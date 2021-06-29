# import create_pretraining_data
import sys
import xml.dom.minidom # from xml.dom import minidom #
import json
import os

import glob
# from transformers import BertTokenizer
import spacy
nlp = spacy.load("en_core_web_sm")

nlp.add_pipe('sentencizer') # updated
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def read_paper_xml(xml_file_path):
	dir_path = os.path.dirname(os.path.realpath(__file__))
	print(dir_path)
	print(xml_file_path)
	cur_xml = xml.dom.minidom.parse(dir_path+xml_file_path)  # the xml_file_path
	cur_XML_root = cur_xml.documentElement
	xml_paragraphs = cur_XML_root.getElementsByTagName('p')  # p
	full_texts = []
	# nlp(xml_paragraphs[0].firstChild.data)
	# full_texts = [str(sent) for sent in full_texts.sents]
	for cur_paragraph in xml_paragraphs:
		full_texts = full_texts + [str(sent) for sent in nlp(cur_paragraph.firstChild.data).sents]
	output_full_texts = "\n".join(full_texts)
	if not os.path.exists('output_txt'):
		os.makedirs('output_txt')
	with open('output_txt/xml_file_path.txt', 'w') as f:
		f.write(output_full_texts)


if __name__ == '__main__':
	arguments = sys.argv[1:]
	for arg in arguments:
		read_paper_xml(arg)