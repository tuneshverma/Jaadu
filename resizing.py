

from PIL import Image
from pytesseract import pytesseract as pt
import argparse





def resizing(image_path):
    
    im = Image.open(image_path)
    nx, ny = im.size
    im = im.resize((int(nx*4.5), int(ny*4.5)), Image.BICUBIC)
    im.save( image_path.split('.')[0]+ '_resized.jpg' , dpi=(200,200))



# def main()

# image_path = '/home/sasuke/Downloads/All_detection/python/cropped_images/do.jpg'





# resizing(image_path)


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--path', type = str , help='Image path', required=True)
	args = parser.parse_args()
	d = vars(args)
	# print(d['path'])
	# print(type(vars(args)))
	resizing(d['path'])

	

if __name__ == "__main__":
	main()



 	



