# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import codecs


def read_document(text):
    # each paragraph is a seperate document? prob - cuz many are very long
    
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {text}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_paragraph(text):
    #which number paragraph of which doc
    #text and make some embedding


# Press the green button in the gutter to run the script.
# f = open("C:/Users/ysnow/OneDrive/Desktop/summer/responsa_for_research/0000000", "rb")
# print(f.read())
# text = f.read()
# text1 = text.decode("iso-8859-8")
# print("\n")
# print(text1)
# text2 = text.decode("cp1255")
# print("\n")
# print(text2)
# text3 = text.decode("cp862")
# print("\n")
# print(text3)
# print_file('PyCharm')
encoding = "cp1255"
f = codecs.open("C:/Users/ysnow/OneDrive/Desktop/summer/responsa_for_research/0000000", "r", encoding)
print(f.read())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
