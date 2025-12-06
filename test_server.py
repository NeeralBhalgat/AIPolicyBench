import asyncio
import json
from src.utils import a2a_client
from a2a.types import SendMessageSuccessResponse, Message
from a2a.utils import get_text_parts

async def test_agent():
    url = "http://localhost:9002"
    query = "What are the three pillars of America's AI Action Plan?"
    
    print(f"🚀 Sending query to {url}...")
    print(f"❓ Query: {query}")
    
    try:
        # Use the A2A Client to handle protocol details
        response = await a2a_client.send_message(url, query)
        
        print(f"\n✅ Response Received:")
        
        # Extract text from response
        res_root = response.root
        if isinstance(res_root, SendMessageSuccessResponse):
            res_result = res_root.result
            if isinstance(res_result, Message):
                text_parts = get_text_parts(res_result.parts)
                if text_parts:
                    print(f"\n💬 Answer:\n{text_parts[0]}")
                else:
                    print("⚠️ No text in response")
            else:
                print(f"⚠️ Unexpected response type: {type(res_result)}")
        else:
            print(f"❌ Error: {response}")
            
    except Exception as e:
        print(f"\n❌ Error connecting to agent: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())
