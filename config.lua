local cfg = {
  genres = {'rock','dubstep','pop','reggae','classic'},


  files = 150, -- num slices per genre to use in final dataset
  valRatio = 0, -- validation ratio
  testRatio = 0, -- test ratio
  slicesPerGenre = 30000, -- num slices per genre initially created
  slice = 128, -- slice size

  -- Directories for files
  dir = {
    data = '../Data/', -- original music directory separated by genre
    spec = "./Spectrograms/", -- store initial spectrograms here
    slices = "./Slices/", -- store ~2 second spectrogram slices here
    dataset = "./Dataset/", -- store serialized instances and labels for train and test
    raw = "./Raw/", -- store tmp files
    uji= "./Uji/",
    dataUji="./Uji/Data/"
  },

  -- Spectrogram params
  spect = {
    pps = 50, -- spectrogram resolution (pixels per second)
    sliceSz = 128, -- slice size
    stride = 128,
    window_type = 'rect'
  },

  -- Model Params
  model = {
    batchSize = 1, -- input size: 1 x slice x slice
    lr = 0.0001,
    epochs = 100
  }
}

return cfg
