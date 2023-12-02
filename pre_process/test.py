import io

from fastapi import UploadFile

from langchain_file_spliter.main import upload_docs

if __name__ == "__main__":
    file_path = "./assets/合同履约-合同结算管理操作手册(1).pdf"

    # 读取文件内容
    with open(file_path, "rb") as f:
        file_contents = f.read()
    upload_file = UploadFile(
        file=io.BytesIO(file_contents), filename="合同履约-合同结算管理操作手册(1).pdf"
    )

    upload_docs(
        files=[upload_file],
        knowledge_base_name="assets",
        override=True,
        chunk_size=500,
        chunk_overlap=50,
        zh_title_enhance=True,
        docs={},
    )
