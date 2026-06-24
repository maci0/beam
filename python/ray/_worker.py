"""Actor worker subprocess: ``python -m ray._worker``.

beamd spawns one of these per actor, with CUDA_VISIBLE_DEVICES / BEAM_GPU_IDS /
BEAM_ACTOR_ID set. It attaches to the daemon, instantiates the pickled class,
then serves method calls one at a time (Ray actors are single-threaded).
"""

import os
import socket
import sys
import traceback

from . import _proto


def _reply(sock, req, payload=b"", err=""):
    header = {"t": req["t"] + "_ok", "reqid": req.get("reqid", 0), "resp": True}
    if err:
        header["err"] = err
        payload = b""
    _proto.write_frame(sock, header, payload)


def main():
    sock_path = os.environ["BEAM_SOCK"]
    actor_id = os.environ["BEAM_ACTOR_ID"]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    _proto.write_frame(sock, {"t": "worker_hello", "actor": actor_id})

    instance = None
    while True:
        try:
            header, payload = _proto.read_frame(sock)
        except ConnectionError:
            return
        if header.get("resp"):
            continue  # the worker_hello acknowledgement; nothing to do

        t = header.get("t")
        try:
            if t == "init":
                cls, args, kwargs = _proto.loads(payload)
                instance = cls(*args, **kwargs)
                _reply(sock, header)
            elif t == "method":
                args, kwargs = _proto.loads(payload)
                method = getattr(instance, header["method"])
                result = method(*args, **kwargs)
                _reply(sock, header, _proto.dumps(result))
            else:
                _reply(sock, header, err="unknown worker op: %s" % t)
        except Exception:
            _reply(sock, header, err=traceback.format_exc())
            print("beam worker %s error:\n%s" % (actor_id, traceback.format_exc()), file=sys.stderr)
            if t == "init":
                return  # no instance: exit instead of serving methods on None


if __name__ == "__main__":
    main()
