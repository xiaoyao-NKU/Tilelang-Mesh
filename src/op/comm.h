#ifndef TVM_TL_OP_COMM_H_
#define TVM_TL_OP_COMM_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

TVM_DLL const Op &CoreId();
TVM_DLL const Op &comm_current_core();
TVM_DLL const Op &comm_barrier();
TVM_DLL const Op &comm_fence();
TVM_DLL const Op &broadcast_();

using namespace tir;


class BroadcastNode : public TileOperatorNode {
public:
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  IntImm size;
  IntImm dst_offset;
  IntImm src_core;
  Array<IntImm> group;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_broadcast", BroadcastNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<BroadcastNode>()
        .def_ro("src", &BroadcastNode::src)
        .def_ro("dst", &BroadcastNode::dst)
        .def_ro("src_range", &BroadcastNode::src_range)
        .def_ro("dst_range", &BroadcastNode::dst_range)
        .def_ro("src_core", &BroadcastNode::src_core)
        .def_ro("size", &BroadcastNode::size)
        .def_ro("dst_offset", &BroadcastNode::dst_offset)
        .def_ro("group", &BroadcastNode::group);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
  TileOperator Clone() const;
};

class Broadcast : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Broadcast, TileOperator, BroadcastNode);
  TVM_DLL Broadcast(Array<PrimExpr> args);
  static const Op &Get();
};


class PutNode : public TileOperatorNode {
public:
  PrimExpr src, dst;
  PrimExpr src_core, dst_core;
  IntImm size;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_put", PutNode, TileOperatorNode);

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PutNode>()
        .def_ro("src", &PutNode::src)
        .def_ro("dst", &PutNode::dst)
        .def_ro("src_core", &PutNode::src_core)
        .def_ro("dst_core", &PutNode::dst_core)
        .def_ro("size", &PutNode::size);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
  TileOperator Clone() const;
};

class Put : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Put, TileOperator, PutNode);
  TVM_DLL Put(Array<PrimExpr> args);
  static const Op &Get();
};


class AllgatherNode : public TileOperatorNode {
public:
  PrimExpr send, recv;
  IntImm size;
  Array<IntImm> group;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_allgather", AllgatherNode, TileOperatorNode);
  
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<AllgatherNode>()
        .def_ro("send", &AllgatherNode::send)
        .def_ro("recv", &AllgatherNode::recv)
        .def_ro("size", &AllgatherNode::size)
        .def_ro("group", &AllgatherNode::group);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
  TileOperator Clone() const;
};

class Allgather : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Allgather, TileOperator, AllgatherNode);
  TVM_DLL Allgather(Array<PrimExpr> args);
  static const Op &Get();
};


class ReduceNode : public TileOperatorNode {
public:
  IntImm op;
  Buffer src, dst;
  Array<Range> src_range, dst_range;
  IntImm axis;
  Array<PrimExpr> group;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.comm_reduce", ReduceNode, TileOperatorNode);
  
  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ReduceNode>()
        .def_ro("op", &ReduceNode::op)
        .def_ro("src", &ReduceNode::src)
        .def_ro("dst", &ReduceNode::dst)
        .def_ro("axis", &ReduceNode::axis)
        .def_ro("group", &ReduceNode::group);
  }

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const override;
  TileOperator Clone() const;
};  

class Reduce : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Reduce, TileOperator, ReduceNode);
  TVM_DLL Reduce(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_COMM_H_
