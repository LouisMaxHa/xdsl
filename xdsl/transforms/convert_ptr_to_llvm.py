from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, memref, ptr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.canonicalization_patterns.ptr import RedundantToPtr


@dataclass
class ConvertStoreOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.StoreOp, rewriter: PatternRewriter, /):
        value = op.value
        new_ops: list = []

        # `index` n'est pas un type LLVM : on cast vers i64 avant le store.
        if isinstance(value.type, builtin.IndexType):
            cast_index = arith.IndexCastOp(value, builtin.i64)
            new_ops.append(cast_index)
            value = cast_index.result

        cast_ptr = builtin.UnrealizedConversionCastOp.get(
            (op.addr,), (llvm.LLVMPointerType(),)
        )
        new_ops.extend([cast_ptr, llvm.StoreOp(value, cast_ptr.results[0])])
        rewriter.replace_op(op, new_ops)


@dataclass
class ConvertLoadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter, /):
        is_index = isinstance(op.res.type, builtin.IndexType)
        # `index` n'est pas un type LLVM : on charge en i64 et on re-cast ensuite.
        llvm_type = builtin.i64 if is_index else op.res.type

        cast_ptr = builtin.UnrealizedConversionCastOp.get(
            [op.addr], [llvm.LLVMPointerType()]
        )
        load = llvm.LoadOp(cast_ptr.results[0], llvm_type)

        if is_index:
            cast_index = arith.IndexCastOp(load.dereferenced_value, builtin.IndexType())
            rewriter.replace_op(op, [cast_ptr, load, cast_index])
        else:
            rewriter.replace_op(op, [cast_ptr, load])


@dataclass
class ConvertPtrAddOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.PtrAddOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op,
            (
                cast_addr_op := builtin.UnrealizedConversionCastOp.get(
                    [op.addr],
                    [llvm.LLVMPointerType()],
                ),
                # offset (index) -> offset (int)
                offest_to_int_op := arith.IndexCastOp(op.offset, builtin.i64),
                # ptr -> int
                ptr_to_int_op := llvm.PtrToIntOp(
                    cast_addr_op.results[0],
                    builtin.i64,
                ),
                # int + arg
                add_op := arith.AddiOp(
                    ptr_to_int_op.results[0], offest_to_int_op.result
                ),
                # int -> ptr
                llvm.IntToPtrOp(add_op.result),
            ),
        )


class ConvertToPtrOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.ToPtrOp, rewriter: PatternRewriter, /):
        source = op.source
        source_owner = source.owner

        # Cas fallback : si ConvertFromPtrOp a déjà tourné avant nous, la source
        # est un cast implicite (!llvm.ptr → memref).  On passe directement l'entrée
        # du cast, évitant un extract_aligned_pointer_as_index sur un !llvm.ptr.
        if isinstance(source_owner, builtin.UnrealizedConversionCastOp):
            rewriter.replace_op(op, (), [source_owner.inputs[0]])
            return

        # Les roundtrips from_ptr→to_ptr sont éliminés par RedundantToPtr (qui tourne
        # avant ce pattern dans GreedyRewritePatternApplier).  Si un from_ptr arrive
        # ici malgré tout (ex. ordre d'application inhabituel), on passe pareil.
        if isinstance(source_owner, ptr.FromPtrOp):
            rewriter.replace_op(op, (), [source_owner.operands[0]])
            return

        # Pour un vrai memref (argument de fonction, memref.alloc, memref.view, …),
        # on extrait le pointeur de données depuis le descripteur.
        extract = memref.ExtractAlignedPointerAsIndexOp.get(source)
        cast_i64 = arith.IndexCastOp(extract.aligned_pointer, builtin.i64)
        inttoptr = llvm.IntToPtrOp(cast_i64.result)
        rewriter.replace_op(op, [extract, cast_i64, inttoptr])


class ConvertFromPtrOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.FromPtrOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, (), op.operands)


class RewritePtrTypes(TypeConversionPattern):
    """
    Replaces `ptr_dxdsl.ptr` with `llvm.ptr`.
    """

    @attr_type_rewrite_pattern
    def convert_type(self, typ: ptr.PtrType):
        return llvm.LLVMPointerType()


class ConvertPtrToLLVMPass(ModulePass):
    name = "convert-ptr-to-llvm"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # Première passe : éliminer tous les roundtrips from_ptr→to_ptr pendant que
        # les types sont encore !ptr_xdsl.ptr (avant RewritePtrTypes).  Cela garantit
        # que ConvertToPtrOp ne voit que des vrais memrefs (function args, view, …).
        PatternRewriteWalker(
            GreedyRewritePatternApplier([RedundantToPtr()])
        ).rewrite_module(op)

        # Deuxième passe : conversion ptr_xdsl → llvm proprement dite.
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertStoreOp(),
                    ConvertLoadOp(),
                    ConvertPtrAddOp(),
                    ConvertToPtrOp(),
                    ConvertFromPtrOp(),
                    RewritePtrTypes(recursive=True),
                ]
            )
        ).rewrite_module(op)
