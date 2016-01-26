--[[
 Long Short-Term Memory.
--]]

local CharCNN, parent = torch.class('CharCNN', 'nn.Module')

function CharCNN:__init(config)
  parent.__init(self)

  self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  self.seq_length = 1014
  self.inputFrameSize = #self.alphabet
  self.outputFrameSize = 256
  self.kw = 7
  self.kw2 = 3
  self.dw = 1
  self.pool_kw = 3
  self.pool_dw = 3

  self.reshape_dim = 34 * self.outputFrameSize

  self.cnn_model = self:new_model() 

end

function CharCNN:new_model()

  local cnn = nn.Sequential()
    :add(nn.Transpose({1,2}))

    :add(nn.TemporalConvolution(self.inputFrameSize, self.outputFrameSize, self.kw, self.dw))
    :add(nn.Threshold())
    :add(nn.TemporalMaxPooling(self.pool_kw, self.pool_dw))

    :add(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw, self.dw))
    :add(nn.Threshold())
    :add(nn.TemporalMaxPooling(self.pool_kw, self.pool_dw))

    :add(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw))
    :add(nn.Threshold())

    :add(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw))
    :add(nn.Threshold())

    :add(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw))
    :add(nn.Threshold())

    :add(nn.TemporalConvolution(self.outputFrameSize, self.outputFrameSize, self.kw2, self.dw))
    :add(nn.Threshold())
    :add(nn.TemporalMaxPooling(self.pool_kw, self.pool_dw))

    :add(nn.Reshape(self.reshape_dim))
    :add(nn.Linear(self.reshape_dim, 1024))
    :add(nn.Threshold())
    :add(nn.Dropout(0.5))

    :add(nn.Linear(1024, 1024))
    :add(nn.Threshold())
    :add(nn.Dropout(0.5))

  local input = nn.Identity()()
  local output = cnn(input)

  local graph = nn.gModule({input}, {output})

  -- share parameters
  if self.cnn_model then
    share_params(graph, self.cnn_model)
  end
  return graph

end

function CharCNN:forward(input)
  self.output = self.cnn_model:forward(input)
  return self.output
end

function CharCNN:backward(input, gradOutput)
  self.gradInput = self.cnn_model:backward(input, gradOutput)
  return self.gradInput
end

function CharCNN:share(cnn, ...)
  share_params(self.cnn_model, cnn.cnn_model, ...)
end

function CharCNN:zeroGradParameters()
  self.cnn_model:zeroGradParameters()
end

function LSTM:parameters()
  return self.cnn_model:parameters()
end