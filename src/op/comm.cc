
#include "comm.h"
#include "../layout/tcgen05_layout.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/common/loop_parallel_transform_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "utils.h"

#include "../target/cuda.h"
#include "../target/utils.h"
#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

#define TIR_DEFINE_TL_BUILTIN(OpName)                                          \
  const Op &OpName() {                                                         \
    static const Op &op = Op::Get("tl." #OpName);                              \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl." #OpName)                                               \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)
TIR_DEFINE_TL_BUILTIN(comm_barrier)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(comm_fence)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(CoreId).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(comm_current_core)
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
TIR_DEFINE_TL_BUILTIN(broadcast_)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));
// src_buffer, dst_buffer, src_core, size, direction(1: horizontal, 2:
// vertical), *group

using namespace tir;

Broadcast::Broadcast(Array<PrimExpr> args) {
  ObjectPtr<BroadcastNode> node = tvm::ffi::make_object<BroadcastNode>();
  Array<Range> rgs[2];
  Buffer bf[2];
  for (int i = 0; i < 2; i++) {
    auto region = NormalizeToBufferRegion(args[i]);
    rgs[i] = region->region;
    bf[i] = region->buffer;
  }
  std::tie(node->src, node->dst) = std::tie(bf[0], bf[1]);
  std::tie(node->src_range, node->dst_range) = std::tie(rgs[0], rgs[1]);
  node->size = Downcast<IntImm>(args[2]);
  node->dst_offset = Downcast<IntImm>(args[3]);
  node->src_core = Downcast<IntImm>(args[4]);
  for (size_t i = 5; i < args.size(); i++) {
    node->group.push_back(Downcast<IntImm>(args[i]));
  }
  data_ = std::move(node);
}

TileOperator BroadcastNode::Clone() const {
  auto op = tvm::ffi::make_object<BroadcastNode>(*this);
  return Broadcast(op);
}

LayoutMap BroadcastNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  return {};
}

Stmt BroadcastNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  ICHECK(target->GetTargetDeviceType() == kDLSUNMMIO)
      << "Broadcast only supports SUNMMIO targets.";
  int mesh_x = target->GetAttr<Integer>("mesh_shape_x").value()->value;
  int mesh_y = target->GetAttr<Integer>("mesh_shape_y").value()->value;

  ICHECK(src_core->value >= 0 and src_core->value < mesh_x * mesh_y)
      << "Source core id " << src_core->value << " out of range [0, "
      << mesh_x * mesh_y << ")";
  for (size_t i = 0; i < group.size(); i++) {
    ICHECK(group[i]->value >= 0 and group[i]->value < mesh_x * mesh_y)
        << "Group core id " << group[i]->value << " out of range [0, "
        << mesh_x * mesh_y << ")";
  }

  PrimExpr src_elements = 1;
  for (size_t i = 0; i < src_range.size(); i++) {
    src_elements *= src_range[i]->extent;
  }
  src_elements = analyzer->Simplify(src_elements);
  PrimExpr dst_elements = 1;
  for (size_t i = 0; i < dst_range.size(); i++) {
    dst_elements *= dst_range[i]->extent;
  }
  dst_elements = analyzer->Simplify(dst_elements);
  ICHECK(Downcast<IntImm>(src_elements)->value <=
         Downcast<IntImm>(dst_elements)->value)
      << "Source buffer size larger than destination buffer size: "
      << src_elements << " vs " << dst_elements;
  ICHECK(size->value <= Downcast<IntImm>(src_elements)->value)
      << "Broadcast size larger than data size: " << size->value << " vs "
      << Downcast<IntImm>(src_elements)->value;

  PrimExpr broadcast_elements;
  if (size->value < 0) {
    broadcast_elements = src_elements;
  } else {
    broadcast_elements = size;
  }
  ICHECK((Downcast<IntImm>(broadcast_elements)->value + dst_offset->value) <=
         Downcast<IntImm>(dst_elements)->value)
      << "Broadcast size + dst_offset larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value + dst_offset->value)
      << " vs " << Downcast<IntImm>(dst_elements)->value;

  PrimExpr src_addr = src.access_ptr(1, DataType::Handle(), 1, 0, src_elements);
  PrimExpr dst_addr =
      dst.access_ptr(2, DataType::Handle(), 1,
                     Downcast<IntImm>(dst_offset->value), src_elements);

  int src_core_x = src_core->value / mesh_y;
  int src_core_y = src_core->value % mesh_y;

  Array<Array<int>> group_mesh;
  for (size_t i = 0; i < mesh_x; i++) {
    group_mesh.push_back(Array<int>());
  }
  for (size_t i = 0; i < group.size(); i++) {
    int row = group[i]->value / mesh_y;
    int col = group[i]->value % mesh_y;
    Array<int> tmp = group_mesh[row];
    tmp.push_back(col);
    group_mesh.Set(row, tmp);
  }
  Array<int> group_rows;
  for (size_t i = 0; i < mesh_x; i++) {
    if (group_mesh[i].size() > 0) {
      group_rows.push_back(i);
    }
  }

  if (group_rows.size() > 1 or group_rows[0] != src_core_x) {
    // 2D broadcast via row-wise and column-wise broadcasts
    Array<Stmt> seq;
    // Row-wise broadcast
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(broadcast_elements);
    args.push_back(src_core);
    args.push_back(2); // direction: row-wise
    for (const auto &r : group_rows) {
      args.push_back(IntImm(DataType::Int(32), r));
    }
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    seq.push_back(broadcast);
    // Column-wise broadcast
    for (size_t i = 0; i < group_rows.size(); i++) {
      int row = group_rows[i];
      if (group_mesh[row].size() == 1 and group_mesh[row][0] == src_core_y) {
        continue;
      }
      Array<PrimExpr> args;
      args.push_back(src_addr);
      args.push_back(dst_addr);
      args.push_back(broadcast_elements);
      args.push_back(int(row * mesh_y) + src_core_y);
      args.push_back(1); // direction: column-wise
      for (size_t j = 0; j < group_mesh[row].size(); j++) {
        args.push_back(IntImm(DataType::Int(32), group_mesh[row][j]));
      }
      Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
      seq.push_back(broadcast);
    }
    Stmt seqstmt = SeqStmt::Flatten(seq);
    return seqstmt;
  } else {
    // 1D broadcast
    Array<PrimExpr> args;
    args.push_back(src_addr);
    args.push_back(dst_addr);
    args.push_back(broadcast_elements);
    args.push_back(src_core);
    args.push_back(1); // direction: column-wise
    for (const auto &g : group_mesh[src_core_x]) {
      args.push_back(IntImm(DataType::Int(32), g));
    }
    Stmt broadcast = Evaluate(Call(DataType::Handle(), broadcast_(), args));
    return broadcast;
  }
  // Stmt reduce = Evaluate(
  //     tvm::tir::Call(DataType::Handle(), tvm::tl::loop_break(), {}));
  // return reduce;
}

TIR_REGISTER_TL_TILE_OP(Broadcast, comm_broadcast)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

Put::Put(Array<PrimExpr> args) {
  ObjectPtr<PutNode> node = tvm::ffi::make_object<PutNode>();
  node->src = args[0];
  node->dst = args[1];
  node->size = Downcast<IntImm>(args[2]);
  node->src_core = args[3];
  node->dst_core = args[4];
  data_ = std::move(node);
}

TileOperator PutNode::Clone() const {
  auto op = tvm::ffi::make_object<PutNode>(*this);
  return Put(op);
}

LayoutMap PutNode::InferLayout(const LayoutInferArgs &T,
                               InferLevel level) const {
  return {};
}

Stmt PutNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Broadcast bcast = Broadcast({
      src,
      dst,
      size,
      0,
      src_core,
      dst_core,
  });
  return bcast->Lower(T, analyzer);
}

TIR_REGISTER_TL_TILE_OP(Put, comm_put)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

Allgather::Allgather(Array<PrimExpr> args) {
  ObjectPtr<AllgatherNode> node = tvm::ffi::make_object<AllgatherNode>();
  node->send = args[0];
  node->recv = args[1];
  node->size = Downcast<IntImm>(args[2]);
  for (size_t i = 3; i < args.size(); i++) {
    node->group.push_back(Downcast<IntImm>(args[i]));
  }
  data_ = std::move(node);
}
TileOperator AllgatherNode::Clone() const {
  auto op = tvm::ffi::make_object<AllgatherNode>(*this);
  return Allgather(op);
}

LayoutMap AllgatherNode::InferLayout(const LayoutInferArgs &T,
                                     InferLevel level) const {
  return {};
}

Stmt AllgatherNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Array<Stmt> bcast_stmts;

  Array<Range> send_range, recv_range;
  auto send_region = NormalizeToBufferRegion(send);
  auto recv_region = NormalizeToBufferRegion(recv);
  send_range = send_region->region;
  recv_range = recv_region->region;
  PrimExpr stride = 1;
  for (size_t i = 0; i < send_range.size(); i++) {
    stride *= send_range[i]->extent;
  }
  stride = analyzer->Simplify(stride);
  PrimExpr send_elements = 1;
  for (size_t i = 0; i < send_range.size(); i++) {
    send_elements *= send_range[i]->extent;
  }
  send_elements = analyzer->Simplify(send_elements);
  PrimExpr recv_elements = 1;
  for (size_t i = 0; i < recv_range.size(); i++) {
    recv_elements *= recv_range[i]->extent;
  }
  recv_elements = analyzer->Simplify(recv_elements);
  ICHECK(Downcast<IntImm>(send_elements)->value * group.size() <=
         Downcast<IntImm>(recv_elements)->value)
      << "Receive buffer size not enough for allgather: required "
      << (Downcast<IntImm>(send_elements)->value * group.size()) << ", but got "
      << Downcast<IntImm>(recv_elements)->value;

  IntImm src_core;
  for (size_t i = 0; i < group.size(); i++) {
    src_core = group[i];

    Array<PrimExpr> args;
    args.push_back(send);
    args.push_back(recv);
    args.push_back(size);
    args.push_back(IntImm(DataType::Int(32), i) * stride);
    args.push_back(src_core);
    for (size_t j = 0; j < group.size(); j++) {
      args.push_back(group[j]);
    }
    Broadcast bcast = Broadcast(args);
    Stmt bcast_stmt = bcast->Lower(T, analyzer);
    bcast_stmts.push_back(bcast_stmt);
  }
  return SeqStmt::Flatten(bcast_stmts);
}

TIR_REGISTER_TL_TILE_OP(Allgather, comm_allgather)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

Reduce::Reduce(Array<PrimExpr> args) {
  ObjectPtr<ReduceNode> node = tvm::ffi::make_object<ReduceNode>();
  node->op = Downcast<IntImm>(args[0]);
  Array<Range> src_rgs, dst_rgs;
  Buffer src_bf, dst_bf;
  auto src_region = NormalizeToBufferRegion(args[1]);
  src_rgs = src_region->region;
  src_bf = src_region->buffer;
  auto dst_region = NormalizeToBufferRegion(args[2]);
  dst_rgs = dst_region->region;
  dst_bf = dst_region->buffer;
  node->src = src_bf;
  node->src_range = src_rgs;
  node->dst = dst_bf;
  node->dst_range = dst_rgs;
  node->axis = Downcast<IntImm>(args[3]);
  node->group = Array<PrimExpr>();
  for (size_t i = 4; i < args.size(); i++) {
    node->group.push_back(args[i]);
  }
  data_ = std::move(node);
}

TileOperator ReduceNode::Clone() const {
  auto op = tvm::ffi::make_object<ReduceNode>(*this);
  // if (par_op_.defined()) {
  //   op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  // }
  return Reduce(op);
}

LayoutMap ReduceNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  return {};
}

Stmt ReduceNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Stmt reduce =
      Evaluate(tvm::tir::Call(DataType::Handle(), tvm::tl::loop_break(), {}));
  return reduce;
}
TIR_REGISTER_TL_TILE_OP(Reduce, comm_reduce)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  PutNode::RegisterReflection();
  BroadcastNode::RegisterReflection();
  AllgatherNode::RegisterReflection();
  ReduceNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
