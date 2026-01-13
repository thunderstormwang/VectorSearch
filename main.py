# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import asyncio


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


async def main():
    print_hi('PyCharm')

    from uat_product_to_milvus import process_and_store_all, process_and_store_all_async, search_milvus_async
    process_and_store_all()
    await search_milvus_async(22335, 20)

    # asyncio.run(batch_process_and_store_all())
    await process_and_store_all_async()
    await search_milvus_async(22335, 20)

    # from debug_milvus import run_debug
    # run_debug(22335)

    # from vertex_embed_and_store import (vertex_main, search_milvus, search_many_texts, query_by_path)
    # vertex_main()
    # # search_milvus(text="紅色跑車")
    # search_many_texts(["紅色跑車", "藍色跑車", "狗", "貓", "人"])
    # query_by_path("./images\鞋貓劍客.jpg")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())