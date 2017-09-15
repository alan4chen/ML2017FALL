from PIL import Image
import sys

im = Image.open(sys.argv[1])
width, height = im.size

rgb_im = im.convert('RGB')
for i in range(width):
	for j in range(height):
		r, g, b = im.getpixel((i,j))
		rgb_im.putpixel((i, j), (int(r/2), int(g/2), int(b/2)))

fw = open("Q2.jpg", "w")
rgb_im.save(fw, "JPEG")
fw.close()