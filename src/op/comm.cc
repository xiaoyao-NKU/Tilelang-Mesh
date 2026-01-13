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
    TIR_DEFINE_TL_BUILTIN(comm_barrier).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_fence).set_num_inputs(0).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque)); 
    TIR_DEFINE_TL_BUILTIN(CoreId).set_num_inputs(1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(comm_current_core).set_num_inputs(0).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    TIR_DEFINE_TL_BUILTIN(broadcast_).set_num_inputs(-1).set_attr<TCallEffectKind>(
        "TCallEffectKind", Integer(CallEffectKind::kOpaque));
    // args: src_buffer, dst_buffer, src_core, size, direction(1: horizontal, 2: vertical), *group

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
  ICHECK(Downcast<IntImm>(src_elements)->value <= Downcast<IntImm>(dst_elements)->value)
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
  ICHECK((Downcast<IntImm>(broadcast_elements)->value + dst_offset->value) <= Downcast<IntImm>(dst_elements)->value)
      << "Broadcast size + dst_offset larger than destination buffer size: "
      << (Downcast<IntImm>(broadcast_elements)->value + dst_offset->value) << " vs "
      << Downcast<IntImm>(dst_elements)->value;
      
  PrimExpr src_addr = src.access_ptr(1, DataType::Handle(), 1, 0, src_elements);
  PrimExpr dst_addr = dst.access_ptr(2, DataType::Handle(), 1, Downcast<IntImm>(dst_offset->value), src_elements);
  
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
    args.push_back(2);  // direction: row-wise
    for (const auto& r : group_rows) {
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
      args.push_back(1);  // direction: column-wise
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
    args.push_back(1);  // direction: column-wise
    for (const auto& g : group_mesh[src_core_x]) {
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


TVM_FFI_STATIC_INIT_BLOCK() {
  BroadcastNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
