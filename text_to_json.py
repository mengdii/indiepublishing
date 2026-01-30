# Save content summaries of publications from a txt file to a json file

import json

books_dataset = []
years = []

with open("catalog_data_unified.txt") as new_file:
    for line in new_file:

        line = line.replace("\n", "")
        line = line.replace("\t", "")
        line = line.replace("'s", "")
        line = line.replace("\u2014", "-")
        line = line.replace("\u2019s", "")
        line = line.replace("\u201c", "")
        line = line.replace("\u201d", "")

        # Create a dictionary
        book = {}
        year = ""

        i = 0
        while i < len(line):
            i += 1
            if i == 4:
                break

        year = line[0:i]
        book[year] = line[i:]
        books_dataset.append(book)


with open("book_catalog.json", "w") as outfile:
    json.dump(books_dataset, outfile)
