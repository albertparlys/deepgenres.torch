---------------------------------TEST PROGRAM------------------------------------------
---------------------------------------------------------------------------------------
-- PROGRAM INI UNTUK MELAKUKAN PENGUJIAN PADA SEBUAH LAGU
-- LAGU AKAN DIKLASIFIKASIKAN DENGAN MODEL YANG SUDAH DILATIH SEBELUMNYA
-- SLICE LAGU YANG DI UJI ADALAH 5 SLICE SEBESAR 128x128 PIXELs
-- PENENTUAN GENRE LAGU BERDASARKAN PADA JMLAH KLASIFIKASI TERBESAR TERHADAP 5 SLICE
--
--
-- 1. MENENTUKAN OPSI BERUPA: NAMA FILE UJI, MODEL TERLATIH, SLICE YANG DIDETEKSI.
-- 2. LOAD FILE UJI, SPEKTOGRAM, SLICE, DAN SIMPAN PADA DATASET DGN KETENTUAN.
-- 		- FILE DILOAD KEMUDIAN DI AMBIL CITRA SPEKTOGRAM DENGAN NAMA: uji.png.
--		- FILE DI SLICE PER 128x128p DENGAN NAMA: uji_n.png, dimana n adalah angka
--		  urutan slice mulai dari awal lagu.
-- 3. LOAD PARAMETER YAITU DENGAN MELAKUKAN LOAD PADA MODEL YG MEMILIKI TINGKAT
--	  PENGENALAN TERBAIK BERDASARKAN GLOBAL CORRECT (.t7).
-- 4. MEMILIH SLICE YANG AKAN DIUJI, BISA DARI DEPAN, TGH, ATAU BELAKANG. NAMUN HARUS 
--	  URUT. KEMUDIAN MEMBUAT DATASET DARI SLICE TERSEBUT
-- 5. MEMASUKAN TIAP SLICE PADA DATASET KEDALAM JARINGAN.
-- 6. MENCATAT HASIL TIAP SLICE PADA SUATU PAPAN PENILAIAN (SCORE= {slice1,slice2,slice3,slice4,slice5})
-- 7. MENENTUKAN NILAI TERBANYAK SEHINGGA MENDAPAT KEPUTUSAN.
--
---------------------------------------------------------------------------------------

-- load library
require 'torch'
require 'nn'
require 'pretty-nn'
require 'paths'
require 'os'
require 'image'
require 'audio'
require 'loader'
require 'testFunc'
require 'utils'
require 'optim'
require 'cudnn'
require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
-------------------------

-- 1. MENENTUKAN OPSI BERUPA: NAMA FILE UJI, MODEL TERLATIH, SLICE YANG DIDETEKSI.
local cmd = torch.CmdLine()
cmd:option('-mode', 'test', 'Test mode')
cmd:option('-model', 'sgd3x3', 'Path to pre-trained model to load (sgd3x3, sgd5x5, sgd7x7, adadelta, adagrad, adam)')
cmd:option('-modelType','SGD 3x3','Model type to load')
cmd:option('-songName', '', 'Path to song to load (for manual input song type here)')
cmd:option('-songGenre','classic','Song Genre to test (input manual if songName is manually typed)')
cmd:option('-testSet','1','Use Test set (1-5)')
cmd:option('-backend', 'nn', 'Set to cudnn to use GPU')
cmd:option('-config', 'config.lua', 'Configuration file containing architecture params')
cmd:option('-autoMode', false,'Auto mode for test (true for auto mode, false for manual mode)')
cmd:text()
local opt = cmd:parse(arg)

if opt.model == 'sgd3x3' then
modelName = 'FIXSGD3X3_epochmodel_150.t7'
elseif opt.model == 'sgd5x5' then
modelName = '5X5FIXepochmodel.t7'
elseif opt.model == 'sgd7x7' then
modelName = '7X7FIXepochmodel.t7'
elseif opt.model == 'adadelta' then
modelName = 'FIXADADELTAepochmodel.t7'
elseif opt.model == 'adagrad' then
modelName = 'FIXADAGRADepochmodel.t7'
elseif opt.model == 'adam' then
modelName = 'FIXADAM_150.t7'
else
print("wrong model!")
end

if opt.songName == '' then
if opt.testSet == '1' then
songName = 'ONEOKROCKchannel - ONE OK ROCK 「アンサイズニア」.mp3'
songGenre = 'rock'
elseif opt.testSet == '2' then
songName = 'Firepower Records - Hi Im Ghost - Halfway [Your EDM Exclusive Premiere].mp3'
songGenre = 'dubstep'
elseif opt.testSet == '3' then
songName = 'ciponx - Killing Me Inside - Jangan Pergi Feat.Tiffany.mp3'
songGenre = 'pop'
elseif opt.testSet == '4' then
songName = 'Dubbest - Leaving.mp3'
songGenre = 'reggae'
elseif opt.testSet == '5' then
songName = 'Charles_A_McGraw - Clarinet Quintet- III. Lento - IV. Allegro.mp3'
songGenre = 'classic'
else
end
else
songName = opt.songName
songGenre = opt.songGenre
end

if not opt.autoMode then
print("===========================================================================================================")
print("===========================================================================================================")
print("PROGRAM PENGUJIAN SISTEM PENGGOLONGAN GENRE BERDASARKAN SPEKTOGRAM MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK")
print("===========================================================================================================")
print("===========================================================================================================\n")
print(string.format("Menampilkan proses pengujian pada lagu %s...\n\n",songName))
else
print(opt)
end
local cfg = paths.dofile(opt.config)

function test(songName,genre)
--local sliceFile = paths.dofile('testFunc.lua').sliceFile

-- 2. LOAD FILE UJI, SPEKTOGRAM, SLICE, DAN SIMPAN PADA DATASET DGN KETENTUAN.
-- 		- FILE DILOAD KEMUDIAN DI AMBIL CITRA SPEKTOGRAM DENGAN NAMA: uji.png.
--		- FILE DI SLICE PER 128x128p DENGAN NAMA: uji_n.png, dimana n adalah angka
--		  urutan slice mulai dari awal lagu.
if not opt.autoMode then
print("1. LOAD FILE UJI, SPEKTOGRAM, SLICE, DAN SIMPAN PADA DATASET DGN KETENTUAN.\n")
print(string.format("Membuat spektogram dari lagu %s dengan genre %s...",songName,genre))
c = createspectrogram(songName,genre)
print("Menyimpan Spektogram pada foler Uji/slices/")
print("Memotong Spektogram per 128x128p pada folder: Uji/png/")
number = sliceFile(c,cfg.slice)
print("Menyimpan Potongan Citra pada foler: Uji/slices/")
else
c = createspectrogram(songName,genre)
number = sliceFile(c,cfg.slice)
end

-- 3. LOAD PARAMETER YAITU DENGAN MELAKUKAN LOAD PADA MODEL YG MEMILIKI TINGKAT
--	  PENGENALAN TERBAIK BERDASARKAN GLOBAL CORRECT (.t7).
--backend = nn
if not opt.autoMode then
print("\n\n2. MEMUAT MODEL CNN TERLATIH\n")
print(string.format("Memuat model %s...",modelName))
else
end
local model = torch.load("./Uji/models/"..modelName)
--cudnn.convert(model,nn)
model:remove() -- replace LogSoftmax with Softmax
model:add(nn.SoftMax())
model:evaluate()
model:cuda()
if not opt.autoMode then
print(model)
else
end


-- 4. MEMILIH SLICE YANG AKAN DIUJI, BISA DARI DEPAN, TGH, ATAU BELAKANG. NAMUN HARUS 
--	  URUT. KEMUDIAN MEMBUAT DATASET DARI SLICE TERSEBUT.
local utils = paths.dofile('utils.lua')
local dataUji = torch.CudaTensor(numSamples,128,128)
local labelsUji = torch.CudaTensor(numSamples):zero()
local slicePath = cfg.dir.uji.."slices/"
--print(slicePath)
if not opt.autoMode then
print("\n\n3. MEMBUAT BASIS DATA UJI\n")
else
end
idx=1
filenames = utils.slice(paths.files(slicePath,".png"),1,numSamples,1)
if not opt.autoMode then
print("Memuat Potongan Citra pada folder: Uji/slices/...")
for i = 1, #filenames do
xlua.progress(i,#filenames)
end
else 
end
if opt.songName == '' then
label = utils.getLabel(genre)
else
label = torch.random(1,5)
end
--print(label)
for i,fn in pairs(filenames) do
	img = utils.getImageData(slicePath..fn, 128)
	dataUji[idx] = img
	labelsUji[idx] = label
	idx=idx +1
end
test_X = dataUji
test_Y = labelsUji
local test = {test_X,test_Y}
if not opt.autoMode then
print(string.format("Memuat Basis Data dengan Data Uji %d..",#filenames))
--print(#test_X)
--print(#test_Y)
print("Menyimpan pada folder: Uji/dataset/ dengan nama file data_Uji.t7")
else
end
torch.save(cfg.dir.uji.."dataset/data_Uji.t7",test)
test_data = torch.load(cfg.dir.uji.."dataset/data_Uji.t7")
local X = test_data[1]
local y = test_data[2]
X:cuda()
y:cuda()
--print(X[1])

-- 5. MEMASUKAN TIAP SLICE PADA DATASET KEDALAM JARINGAN.
-- 6. MENCATAT HASIL TIAP SLICE PADA SUATU PAPAN PENILAIAN (SCORE= {slice1,slice2,slice3,slice4,slice5})
if not opt.autoMode then
print("\n\n4. MEMUAT BASIS DATA DAN MENGUJI BASIS DATA\n")
print("Memuat basis data pada folder: Uji/dataset/ dengan nama file data_Uji.t7")
else
end
local num = X:size(1)
correct = num
--print(num)
scores = {0,0,0,0,0}
--scores:cuda()
--print("scores")
--print(scores)
totals = {0,0,0,0,0}
--totals:cuda()
--print("totals")
--print(totals)
confusion = optim.ConfusionMatrix(cfg.genres)
if not opt.autoMode then
print("Pengujian basis data..")
print(string.format("%-10s\t%-10s \t\t    %8s","(No.)","Nama File","Penggolongan"))
else
end
for i = 1, X:size(1) do
--local output = model:forward(X[i]:view(1,X[i]:size(1),X[i]:size(1)))

inputX = X[i]:clone()

input = inputX:view(1,X[i]:size(1),X[i]:size(2))
input:cuda()

output = model:forward(input)
output:cuda()
--print(output)
k, class = torch.max(output,1)
--print(class)
--print(desiredOutput)
--print("class")
--print(class)
confusion:add(class[1],y[i])
totals[class[1]] = totals[class[1]]+1
--    if class[1] ~= y[i] then
--      scores[class[1]] = scores[class[1]] + 1
--      correct = correct-1
--     print(correct)
 --   end
if not opt.autoMode then
print(string.format("(%-d)\t   %-10s \t\t       %-8s",i,filenames[i],cfg.genres[class[1]]))
else
end
end
--print(totals)
genreLabel = descision(totals)
if opt.autoMode then
local utut = io.open(string.format("./Uji/report/FIX/%s_truefalse.txt",opt.modelType) , 'a+') 
if cfg.genres[genreLabel] == genre then
print("(TRUE!) '"..songName.."' detected as "..genre)
file:write(string.format("(TRUE!) '%s' detected as %s\n",songName,string.upper(genre)))
utut:write(string.format("Y\n"))
elseif genreLabel == 0 then
print("(ERROR!) Double genre detected")
else
print(string.format("(WRONG!) '%s' detected as %s (%s) to (%s)",songName,cfg.genres[genreLabel],totals[genreLabel],totals[y[1]]))
file:write(string.format("(WRONG!) '%s' detected as %s (%s) to (%s)\n",songName,cfg.genres[genreLabel],totals[genreLabel],totals[y[1]]))
utut:write(string.format("W\n"))
end
utut:close()
else
end
if not opt.autoMode then
print("\n\nGolongan Genre")
print(cfg.genres)
print("\nHasil Penggolongan")
print(totals)
print("\nKesimpulan penggolongan:")
	if cfg.genres[genreLabel] == genre then
	print("(BENAR!) '"..songName.."' detected as "..genre)
	elseif genreLabel == 0 then
	print("(ERROR!) Double genre detected")
	else
	print(string.format("(SALAH!) '%s' detected as %s (%s) to (%s)\n",songName,cfg.genres[genreLabel],totals[genreLabel],totals[y[1]]))
	end
else 
end
--t= image.display(confusion:render())
--print(confusion)
--image.save(cfg.dir.uji.."report/ujix.jpg",t)
if not opt.autoMode then
print("\n\n===========================================================================================================")
print("===========================================================================================================")
else
end
end

if opt.autoMode then
processDir(cfg,opt.modelType)
else
test(songName, songGenre)
end
