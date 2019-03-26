import os
def data_from_github():
    repo_https_addr = "https://github.com/MichaelYxWang/Stanford_CS231N.git"
    specific_path = "Papers/*"

    os.system("git init")
    os.system("git config core.sparseCheckout true")
    os.system("git remote add -f origin {}".format(repo_https_addr))
    os.system("echo {}  > .git/info/sparse-checkout".format(specific_path))
    os.system("git checkout master")
