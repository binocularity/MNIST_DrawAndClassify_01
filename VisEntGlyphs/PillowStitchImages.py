#
# Create the glyphs with key from the raw Blender image files.
#

from PIL import Image, ImageFont, ImageDraw

fileA = ".\VisEntGlyphs\A_crp.png"
fileB = ".\VisEntGlyphs\B_crp.png"
fileC = ".\VisEntGlyphs\C_crp.png"
fileD = ".\VisEntGlyphs\D_crp.png"
fileE = ".\VisEntGlyphs\E_crp.png"
fileF = ".\VisEntGlyphs\F_crp.png"
fileG = ".\VisEntGlyphs\G_crp.png"


srcA = Image.open(fileA).convert("RGBA")
srcB = Image.open(fileB).convert("RGBA")
srcC = Image.open(fileC).convert("RGBA")
srcD = Image.open(fileD).convert("RGBA")
srcE = Image.open(fileE).convert("RGBA")
srcF = Image.open(fileF).convert("RGBA")
srcG = Image.open(fileG).convert("RGBA")


result = Image.new("RGB", (640*7,640) )

result.paste( srcA, ( 0,0,   640,640 ) )
result.paste( srcB, ( 640,0, 640+640,640 ) )
result.paste( srcC, ( 640*2,0, 640+640*2,640 ) )
result.paste( srcD, ( 640*3,0, 640+640*3,640 ) )
result.paste( srcE, ( 640*4,0, 640+640*4,640 ) )
result.paste( srcF, ( 640*5,0, 640+640*5,640 ) )
result.paste( srcG, ( 640*6,0, 640+640*6,640 ) )

result.save(".\VisEntGlyphs\stitchedGlyphs.png")

img = result.resize((640, 92), Image.ANTIALIAS)
img = img.crop((0,0,640,80))
img.save(".\VisEntGlyphs\stitchedSml.png")

srcA.paste( img, (0,560))
srcA.save(".\VisEntGlyphs\A_crp_over.png")

srcB.paste( img, (0,560))
srcB.save(".\VisEntGlyphs\B_crp_over.png")

srcC.paste( img, (0,560))
srcC.save(".\VisEntGlyphs\C_crp_over.png")

srcD.paste( img, (0,560))
srcD.save(".\VisEntGlyphs\D_crp_over.png")

srcE.paste( img, (0,560))
srcE.save(".\VisEntGlyphs\E_crp_over.png")

srcF.paste( img, (0,560))
srcF.save(".\VisEntGlyphs\F_crp_over.png")

srcG.paste( img, (0,560))
srcG.save(".\VisEntGlyphs\G_crp_over.png")