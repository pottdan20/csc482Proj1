import os
import re

class Document:
    def __init__(self, author, type, docRaw):
        self.author = author
        self.type = type
        self.document = "\n".join(re.findall("<p>(.*?)</p>",docRaw.replace("&nbsp;", "")))

    def __repr__(self):
        return(self.author +" " + self.type+ "\n" )

def preproccess():
    documents=[]
    os.chdir("./proj1F21_files")
    for file in os.listdir():
        with open(file, encoding="utf8") as f:
            docRaw=f.read()
        documents.append(Document(file.split("_")[1],file.split("_")[2].split(".")[0],docRaw))
    return documents 
