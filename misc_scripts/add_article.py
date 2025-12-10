import sys

def add_article(url):
    with open('article_list.txt', 'a') as file:
        file.write('\n' + url)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_article.py <URL>")
    else:
        add_article(sys.argv[1])
