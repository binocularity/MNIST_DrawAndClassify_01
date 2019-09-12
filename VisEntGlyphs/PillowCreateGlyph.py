#
# Test glyph creation for the outcome visual
#

from PIL import Image, ImageFont, ImageDraw

filename = ".\VisEntGlyphs\A_crp.png"

source_img = Image.open(filename).convert("RGBA")

draw = ImageDraw.Draw(source_img)
font = ImageFont.truetype("arial.ttf", 80)
num = "2"
xD,yD=draw.textsize(num, font=font)
draw.text((130-(xD/2),130-(yD/2)), num, fill=(255,255,255,255), font=font )


source_img.save(".\VisEntGlyphs\currentGlyph.png")