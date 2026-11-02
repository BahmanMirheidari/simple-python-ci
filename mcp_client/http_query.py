import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


MCP_URL = "http://localhost:8000/mcp"


async def main():
    question = input("Question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    try:
        async with streamablehttp_client(MCP_URL) as (
            read_stream,
            write_stream,
            get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                print("\nConnected to MCP server.")

                result = await session.call_tool(
                    "ask_question",
                    {"question": question},
                )

                print("\nAnswer:")
                for item in result.content:
                    if hasattr(item, "text"):
                        print(item.text)

                if result.isError:
                    print("\nThe MCP tool returned an error.")

    except Exception as error:
        print(f"\nError: {error!r}")
        if hasattr(error, "exceptions"):
            for i, sub_error in enumerate(error.exceptions, 1):
                print(f"\nUnderlying error {i}:")
                print(repr(sub_error))
                import traceback
                traceback.print_exception(sub_error)
        else:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())