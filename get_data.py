import os
from urllib import request 
from zipfile import ZipFile
from bs4 import BeautifulSoup     
# test   


zip_url = "http://users.csc.calpoly.edu/~foaad/proj1F21_files.zip"
# Download zip file from url
zip_response = request.urlopen(zip_url)
 # Create a new file 
temp_zip = open("./tempfile.zip", "wb")
# Write contents of downloaded zip into new file 
contents = zip_response.read()
temp_zip.write(contents)
temp_zip.close()
# Re-open and extract contents into current directory
in_zip_file = ZipFile("./tempfile.zip")
in_zip_file.extractall(path = "./in_files_html")
os.remove("./tempfile.zip") # remove the zip file
in_zip_file.close()
# Go to directory within created directory (proj1F21_files)
os.chdir("./in_files_html") # go to in_files_html directory 
child_directory = os.listdir()[0] # cd to directory inside it 
os.chdir("./" + child_directory)
# Look at each html file 
for filename in os.listdir(): 
    html = open(filename).read()
    raw = BeautifulSoup(html, 'html.parser').get_text()
    # PROCESS RAW HERE (label and do whatever)
    os.remove(filename) # remove file after done processing       
# clean up other files and directories 
os.chdir("..")
os.rmdir("./" + os.listdir()[0])
os.chdir("..")
os.rmdir("./in_files_html")        