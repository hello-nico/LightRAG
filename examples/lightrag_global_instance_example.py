import asyncio
from lightrag.core.instance_manager import get_lightrag_instance, get_global_manager

async def main():
    rag_instance = await get_lightrag_instance()
    print(rag_instance)
    global_manager = get_global_manager()
    instance_names = await global_manager.get_instance_names()
    print(instance_names)
    
if __name__ == "__main__":
    asyncio.run(main())