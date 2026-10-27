## Setup Environment


```bash

py -3.12 -m venv .venv
python -m venv .venv
.venv\Scripts\activate

python -m pip list
python -m pip install -r .\requirements.txt -U
python -m pip list

deactivate
```




<!-- 

ollama serve
ollama list
ollama pull gemma3:1b
ollama pull qwen3-embedding:0.6b




python -m test.test_real_multi_conversation 
python -m test.test_context 
python -m test.test_real_long_memory_conversation
python -m test.test_fact_extractor
python -m test.test_real_full_conversation
python -m test.test_long_memory_multi_user
python -m test.test_embedding
python -m test.test_embedding_multiple
python -m test.test_embedding_similarity
python -m test.test_rag
python -m test.test_rag_retrieval
python -m test.test_context_with_rag
python -m test.test_real_full_conversation_with_rag
python -m test.test_real_generator_with_mcp
python -m test.test_real_full_conversation_with_mcp
python -m test.test_real_mcp_server
python -m test.test_tool_selection_benchmark
python -m test.test_real_generator_with_mcp
python -m test.test_real_generator_with_mcp
python -m test.test_real_conversation_with_mcp
python -m test.smoke_mem0_pilot
python -m pytest test/test_mem0_isolation.py -v -s
python -u -m pytest test/test_mem0_isolation.py -vv -s --override-ini="python_functions=test_*"


python -m streamlit run ui/streamlit_app.py   
-->




## پاک‌سازی حافظه‌های تستی، بدون حذف Vector Store#
Set-Location "D:\__Personal\Projects\AIAssistant"

# حذف تاریخچه گفتگوها و پیام‌های ذخیره‌شده
Remove-Item ".\data\chat_memory.json" `
    -Force -ErrorAction SilentlyContinue

# حذف واقعیت‌ها و حافظه بلندمدت کاربران آزمایشی
Remove-Item ".\data\long_memory.json" `
    -Force -ErrorAction SilentlyContinue

# اطمینان از باقی ماندن Vector Store
if (Test-Path ".\data\chroma_db") {
    Write-Host "Chroma Vector Store preserved."
} else {
    Write-Warning "Chroma Vector Store not found!"
}

Write-Host "Chat and long-memory cleanup completed."