"""Spin up a basic server endpoint."""

import asyncio
import vizible
from ._src._server import MutableHostedLLM

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                           Constants.                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
_HOST = "127.0.0.1"
_PORT = 8000
_ADDR = f"http://{_HOST}:{_PORT}/static/"


async def main():
    """Main program loop, run asychnronously."""
    server = MutableHostedLLM(host=_HOST, port=_PORT)
    try:
        await server.up()
        vizible.green("Successfully launched server and enter eval framework")
        vizible.blue(f"Server address: \n{_ADDR}")
        await asyncio.sleep(24 * 60 * 60)  # 1 day.
    except KeyboardInterrupt:
        vizible.red("Turn down requested.")
    finally:
        await server.down()
    vizible.green("Async server terminated successfully if you can read this!**")
    vizible.green("\n\n\t\t**(assumes that you have the ability to read)")


if __name__ == "__main__":
    asyncio.run(main())
