# from pdfminer.high_level import extract_text

# text = extract_text("/home/manish/Documents/Manish Internship certificate.pdf")
# print(text)

import PyPDF2

# with open('/home/manish/Documents/Manish Internship certificate.pdf', 'rb') as file:
pdf_reader = PyPDF2.PdfReader('/home/manish/Documents/Manish Internship certificate.pdf')
text = ''
for page_num in range(len(pdf_reader.pages)):
    text += pdf_reader.pages[page_num].extract_text()
