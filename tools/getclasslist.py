def getlist(txt_path):
    with open(txt_path, 'r') as f:
        content = f.read()
        content = content.split('\n')[:-1]
        return content
