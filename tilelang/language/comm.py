"""Communication intrinsics wrappers for TileLang.

This module provides small helper functions that prepare arguments and
emit TIR intrinsics for inter-core communication on a target mesh.
"""

from typing import Tuple, Iterable, Literal

from tvm import tir
import tilelang.language as T
from tilelang.utils.language import to_buffer_region


def CoreId(core_id: int | tuple[int, int]):
    """Convert a core identifier to a linear core ID for the target mesh.

    Parameters
    ----------
    core_id : int or tuple[int, int]
        Either a linear core id (int) or a 2-tuple (row, col) specifying the
        core coordinates on the target mesh.

    Returns
    -------
    int
        The linear core id mapped into [0, mesh_x * mesh_y).

    Raises
    ------
    AssertionError, ValueError
        If the provided coordinates are out of bounds or the type is invalid.
    """
    mesh_shape = T.get_target_mesh_shape("auto")
    if isinstance(core_id, tuple):
        row, col = core_id
        assert (
            0 <= row < mesh_shape["x"]
        ), f"Row {row} out of bounds for mesh shape {mesh_shape}"
        assert (
            0 <= col < mesh_shape["y"]
        ), f"Col {col} out of bounds for mesh shape {mesh_shape}"
        # Convert 2D coordinates into a linear core id.
        core_id_value = row * mesh_shape["x"] + col
    elif isinstance(core_id, int):
        core_id_value = core_id
        assert (
            0 <= core_id_value < mesh_shape["x"] * mesh_shape["y"]
        ), f"Core ID {core_id_value} out of bounds for mesh shape {mesh_shape}"
    else:
        raise ValueError("core_id must be either a tuple[int, int] or an int.")
    return core_id_value


def core_id_to_tuple(core_id: tir.Call) -> Tuple[int, int]:
    """Convert a linear core id into 2D (row, col) coordinates on the mesh.

    Parameters
    ----------
    core_id : tir.Call
        A linear core identifier (or a TIR expression that yields one).

    Returns
    -------
    tuple[int, int]
        The (row, col) coordinates corresponding to the linear core id.

    Notes
    -----
    The conversion uses the current target mesh shape obtained via
    T.get_target_mesh_shape("auto").
    """
    mesh_shape = T.get_target_mesh_shape("auto")
    core_id_value = core_id
    row = core_id_value // mesh_shape["y"]
    col = core_id_value % mesh_shape["y"]
    return (row, col)


def broadcast(
    src: T.Buffer,
    dst: T.Buffer,
    src_core: tuple[int, int],
    group: (
        Literal["horizontal", "h", "vertical", "v", "all", "a"]
        | Iterable[tuple[int, int]]
        | None
    ) = None,
    size: int = -1,
):
    """
    Broadcast data from a source buffer on a specific source core to destination buffers on
    a set of participant cores by emitting the TIR intrinsic tl.tileop.comm_broadcast.

    Parameters
    ----------
    src : T.Buffer
        Source buffer containing data to broadcast.
    dst : T.Buffer
        Destination buffer to receive the broadcasted data.
    src_core : tuple[int, int]
        (row, col) coordinates of the source core on the target mesh.
    group : {'horizontal', 'h', 'vertical', 'v', 'all', 'a'} | iterable of tuple[int, int] | None
        Participant set for the broadcast. Can be one of the following strings:
        - 'horizontal' or 'h': all cores in the same row as the destination core.
        - 'vertical' or 'v': all cores in the same column as the destination core.
        - 'all' or 'a': all cores in the mesh as the destination core.
        Alternatively, an explicit iterable of (row, col) tuples can be provided.
        If None, defaults to 'all'.
    size : int
        Number of elements to broadcast. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_broadcast`.
    Examples
    --------
    >>> broadcast(A, B, (1, 2), group='horizontal')
    >>> broadcast(A, B, (0, 0), group=[(0,0),(0,1),(1,0)])
    """
    assert (
        src.dtype == dst.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {src.dtype} vs {dst.dtype}."
    if len(src.shape) != len(dst.shape):
        raise ValueError(
            "Source and destination buffer must have the same number of dimensions for broadcast."
        )
    for i in range(len(src.shape)):
        assert (
            src.shape[i] == dst.shape[i] or src.shape[i] == 1 or dst.shape[i] == 1
        ), f"Source buffer shape  and destination buffer shape must match for broadcast. Got {src.shape} vs {dst.shape}."

    mesh_shape = T.get_target_mesh_shape("auto")
    assert (
        isinstance(src_core, tuple) and len(src_core) == 2
    ), "src_core must be a tuple of (row, col)."
    assert (
        0 <= src_core[0] < mesh_shape["x"]
    ), f"src_core row {src_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (
        0 <= src_core[1] < mesh_shape["y"]
    ), f"src_core col {src_core[1]} out of bounds for mesh shape {mesh_shape}."

    src_elements = 1
    for dim in src.shape:
        src_elements *= dim
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    assert (
        size <= src_elements
    ), f"size {size} exceeds source buffer size {src_elements}."

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src_core_id = CoreId(src_core)

    if group is None:
        group = "all"
    if isinstance(group, str):
        if group.lower() in ["horizontal", "h"]:
            row, col = core_id_to_tuple(src_core_id)
            group = [(row, c) for c in range(mesh_shape["y"])]
        elif group.lower() in ["vertical", "v"]:
            row, col = core_id_to_tuple(src_core_id)
            group = [(r, col) for r in range(mesh_shape["x"])]
        elif group.lower() in ["all", "a"]:
            group = [
                (r, c) for r in range(mesh_shape["x"]) for c in range(mesh_shape["y"])
            ]
        else:
            raise ValueError(f"Invalid group string: {group}")
    elif isinstance(group, Iterable):
        for core_id in group:
            assert (
                isinstance(core_id, tuple) and len(core_id) == 2
            ), "Each core_id in group must be a tuple of (row, col)."
            assert (
                0 <= core_id[0] < mesh_shape["x"]
            ), f"core_id row {core_id[0]} out of bounds for mesh shape {mesh_shape}."
            assert (
                0 <= core_id[1] < mesh_shape["y"]
            ), f"core_id col {core_id[1]} out of bounds for mesh shape {mesh_shape}."
        pass
    else:
        raise ValueError(
            "group must be either a string or an iterable of tuple[int, int]."
        )

    group = [CoreId(core_id) for core_id in group]
    dst_offset = 0  # Always 0 for now
    args = (src_region, dst_region, size, dst_offset, src_core_id, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_broadcast"), *args)


def put(
    src: T.Buffer,
    dst: T.Buffer,
    src_core: tuple[int, int],
    dst_core: tuple[int, int],
    size: int = -1,
):
    """Put data from a source buffer on a specific source core to a destination buffer on a specific destination core
    by emitting the TIR intrinsic tl.tileop.comm_put.
    Parameters
    ----------
    src : T.Buffer
        Source buffer containing data to put.
    dst : T.Buffer
        Destination buffer to receive the data.
    src_core : tuple[int, int]
        (row, col) coordinates of the source core on the target mesh.
    dst_core : tuple[int, int]
        (row, col) coordinates of the destination core on the target mesh.
    size : int
        Number of elements to put. If -1, the entire source buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_put`.
    Examples
    --------
    >>> put(A, B, (1, 2), (2, 3))
    """
    assert (
        src.dtype == dst.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {src.dtype} vs {dst.dtype}."
    if len(src.shape) != len(dst.shape):
        raise ValueError(
            "Source and destination buffer must have the same number of dimensions for broadcast."
        )
    for i in range(len(src.shape)):
        assert (
            src.shape[i] == dst.shape[i] or src.shape[i] == 1 or dst.shape[i] == 1
        ), f"Source buffer shape  and destination buffer shape must match for broadcast. Got {src.shape} vs {dst.shape}."

    mesh_shape = T.get_target_mesh_shape("auto")
    assert (
        isinstance(src_core, tuple) and len(src_core) == 2
    ), "src_core must be a tuple of (row, col)."
    assert (
        0 <= src_core[0] < mesh_shape["x"]
    ), f"src_core row {src_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (
        0 <= src_core[1] < mesh_shape["y"]
    ), f"src_core col {src_core[1]} out of bounds for mesh shape {mesh_shape}."
    assert (
        isinstance(dst_core, tuple) and len(dst_core) == 2
    ), "dst_core must be a tuple of (row, col)."
    assert (
        0 <= dst_core[0] < mesh_shape["x"]
    ), f"dst_core row {dst_core[0]} out of bounds for mesh shape {mesh_shape}."
    assert (
        0 <= dst_core[1] < mesh_shape["y"]
    ), f"dst_core col {dst_core[1]} out of bounds for mesh shape {mesh_shape}."
    src_elements = 1
    for dim in src.shape:
        src_elements *= dim
    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    assert (
        size <= src_elements
    ), f"size {size} exceeds source buffer size {src_elements}."

    src_region = to_buffer_region(src)
    dst_region = to_buffer_region(dst)
    src_core_id = CoreId(src_core)
    dst_core_id = CoreId(dst_core)
    args = (src_region, dst_region, size, src_core_id, dst_core_id)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_put"), *args)


def all_gather(
    send_buffer: T.Buffer,
    recv_buffer: T.Buffer,
    group: Iterable[tuple[int, int]],
    size: int = -1,
):
    """Gather data from all participant cores into a receive buffer.
    Parameters
    ----------
    send_buffer : T.Buffer
        Source buffer containing local data to send.
    recv_buffer : T.Buffer
        Destination buffer to hold gathered data from all cores.
    group : iterable of tuple[int, int]
        Participant set for the gather operation.
    size : int
        Number of elements to gather from each core. If -1, the entire send buffer is used.
    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.tileop.comm_allgather`.
    Examples
    --------
    >>> all_gather(A, B, group=[(0,0),(0,1),(1,0)])
    """
    mesh_shape = T.get_target_mesh_shape("auto")
    for core_id in group:
        assert (
            isinstance(core_id, tuple) and len(core_id) == 2
        ), "Each core_id in group must be a tuple of (row, col)."
        assert (
            0 <= core_id[0] < mesh_shape["x"]
        ), f"core_id row {core_id[0]} out of bounds for mesh shape {mesh_shape}."
        assert (
            0 <= core_id[1] < mesh_shape["y"]
        ), f"core_id col {core_id[1]} out of bounds for mesh shape {mesh_shape}."

    assert (
        send_buffer.dtype == recv_buffer.dtype
    ), f"Source and destination buffer dtypes must match for broadcast. Got {send_buffer.dtype} vs {recv_buffer.dtype}."
    send_elements = 1
    for dim in send_buffer.shape:
        send_elements *= dim
    recv_elements = 1
    for dim in recv_buffer.shape:
        recv_elements *= dim
    expected_recv_elements = send_elements * len(group)
    assert (
        recv_elements >= expected_recv_elements
    ), f"Receive buffer size {recv_elements} is insufficient to hold gathered data from {len(group)} cores with send buffer size {send_elements}."

    assert isinstance(size, int) and size >= -1, "size must be an integer >= -1."
    assert (
        size <= send_elements
    ), f"size {size} exceeds send buffer size {send_elements}."
    send_buffer_region = to_buffer_region(send_buffer)
    recv_buffer_region = to_buffer_region(recv_buffer)
    group = [CoreId(core_id) for core_id in group]
    args = (send_buffer_region, recv_buffer_region, size, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_allgather"), *args)


def all_reduce(
    op: str,
    src_buffer: T.Buffer,
    dst_buffer: T.Buffer,
    group: Iterable[tuple[int, int]] | None = None,
    axis: int = 0,
):
    """Reduce values across cores using the specified operation.

    Parameters
    ----------
    op : str
        Reduction operation name (for example, 'sum', 'max').
    src_buffer : T.Buffer
        Source buffer containing local values to reduce.
    dst_buffer : T.Buffer
        Destination buffer to hold the reduced result.
    group : iterable of tuple[int, int] | None
        Optional participant set for the reduction.
    axis : int | None
        Optional axis parameter forwarded to the intrinsic if supported.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_reduce`.

    Examples
    --------
    >>> all_reduce('sum', A, B)
    >>> all_reduce('sum', A, B, group=[(0,0),(0,1)], axis=0)
    """
    op_dict = {"sum": 0, "max": 1, "min": 2, "prod": 3}
    assert op in op_dict, f"Reduction op must be one of {op_dict}, but got {op}."
    op = op_dict[op]

    src_buffer_region = to_buffer_region(src_buffer)
    dst_buffer_region = to_buffer_region(dst_buffer)
    if group is None:
        args = (op, src_buffer_region, dst_buffer_region, axis)
    else:
        group = [CoreId(core_id) for core_id in group]
        args = (op, src_buffer_region, dst_buffer_region, axis, *group)
    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.comm_reduce"), *args)


def barrier(group: Iterable[tuple[int, int]] | None = None):
    """Insert a synchronization barrier among a group of cores.

    Parameters
    ----------
    group : iterable of tuple[int, int] | None
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.
        Optional set of core coordinates to synchronize. If omitted, the
        runtime's default participant set is used.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_barrier`.

    Examples
    --------
    >>> barrier()
    >>> barrier(group=[(0,0),(0,1)])
    """
    if group is None:
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"))
    else:
        group = [CoreId(core_id) for core_id in group]
        return tir.call_intrin("handle", tir.op.Op.get("tl.comm_barrier"), *group)


def fence():
    """Emit a memory/communication fence intrinsic.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_fence`.

    Examples
    --------
    >>> fence()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_fence"))


def current_core():
    """Get the current core's identifier.

    Returns
    -------
    tir.Call
        The TIR intrinsic call handle for `tl.comm_current_core`.

    Examples
    --------
    >>> current_core()
    """
    return tir.call_intrin("handle", tir.op.Op.get("tl.comm_current_core"))
