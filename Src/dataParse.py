from llama_index.core import SimpleDirectoryReader
# 加载并解析一个目录下的所有文件
def parse_all_documents(directory_path):
    # 使用 SimpleDirectoryReader 加载目录下的所有文件
    documents = SimpleDirectoryReader(directory_path).load_data()
    # 打印文档信息
    for doc in documents:
        print(f"Document ID: {doc.doc_id}, Content Length: {len(doc.text)},"
              f"file_path: {doc.metadata.get('file_path', 'N/A')}")
    return documents

if __name__ == "__main__":
    # 示例用法
    directory_path = "/Users/drw/PycharmProjects/RAG/Docs" 
    parse_all_documents(directory_path)