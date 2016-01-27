local CharCNNSim_2 = torch.class('CharCNNSim_2')

function CharCNNSim_2:__init(config)

  self.learning_rate = config.learning_rate or 0.05
  self.batch_size    = config.batch_size    or 25
  self.reg           = config.reg           or 1e-4
  self.sim_nhidden   = config.sim_nhidden   or 50

  self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
  self.dict = {}
  for i = 1,#self.alphabet do
    self.dict[self.alphabet:sub(i,i)] = i
  end

  self.length = 600
  self.num_classes = 5

  -- optimizer configuration
  self.optim_state = { learningRate = self.learning_rate, momentum = 0.9, decay = 1e-5 }
  --self.optim_state = { learningRate = self.learning_rate }

  self.criterion = nn.DistKLDivCriterion()

  self.CNN = CharCNN() 
  
  self.params, self.grad_params = self.CNN:getParameters()

  share_params(self.CNN, self.CNN)

end

function CharCNNSim_2:train(dataset)

  self.CNN:training()

  local indices = torch.randperm(dataset.size)
  local avgloss = 0.
  local N = dataset.size / self.batch_size

  for i = 1, dataset.size, self.batch_size do
    xlua.progress(i, dataset.size)
    local batch_size = math.min(i + self.batch_size - 1, dataset.size) - i + 1

    local targets = torch.zeros(batch_size, self.num_classes)
    for j=1, batch_size do
      local sim = dataset.sim_labels[indices[i+j-1]] * (self.num_classes-1)+1
      local ceil, floor = math.ceil(sim), math.floor(sim)

      if ceil ==0 then
        ceil=1
      end
      if floor ==0 then
        floor=1
      end

      if ceil == floor then
        targets[{j, floor}] = 1
      else
        targets[{j, floor}] = ceil -sim
        targets[{j, ceil}] = sim - floor
      end
    end

    local feval = function(x)
      self.grad_params:zero()

      local loss = 0
      for j = 1, batch_size do
        local idx = indices[i + j - 1]
        local lsent, rsent = dataset.lsents[idx], dataset.rsents[idx]

        local linputs = self:seq2vec(lsent)
        local rinputs = self:seq2vec(rsent)

        local inputs = nn.JoinTable(1):forward{linputs,rinputs}
        
        local output = self.CNN:forward(inputs)

        -- compute loss and backpropagate
        local example_loss = self.criterion:forward(output, targets[j])

        loss = loss + example_loss

        local sim_grad = self.criterion:backward(output, targets[j])

        local rep_grad = self.CNN:backward(inputs, sim_grad)

      end

      loss = loss / batch_size
      self.grad_params:div(batch_size)

      -- regularization
      --loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
      avgloss = avgloss + loss
      self.grad_params:add(self.reg, self.params)
      return loss, self.grad_params
    end
    optim.sgd(feval, self.params, self.optim_state)
    avgloss = avgloss/N
  end
  xlua.progress(dataset.size, dataset.size)
  return avgloss
end

-- Predict the similarity of a sentence pair.
function CharCNNSim_2:predict(lsent, rsent)

  self.CNN:evaluate()

  local linputs = self:seq2vec(lsent)
  local rinputs = self:seq2vec(rsent)

  local inputs = nn.JoinTable(1):forward{linputs,rinputs}
        
  local output = self.CNN:forward(inputs)

  self.CNN:forget()

  return torch.range(1,5):dot(output:exp())

end


-- Produce similarity predictions for each sentence pair in the dataset.
function CharCNNSim_2:predict_dataset(dataset)

  local predictions = torch.Tensor(dataset.size)
  for i = 1, dataset.size do
    xlua.progress(i, dataset.size)
    local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
    predictions[i] = self:predict(lsent, rsent)
  end
  return predictions
end

function trim(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function CharCNNSim_2:seq2vec(sequence)

  local s = ''
  --print(sequence)
  for i=1, #sequence do
    s = s .. sequence[i] .. " "
  end
  s = trim(s)
  local s = s:lower()
  
  local t = torch.Tensor(#self.alphabet, self.length)
  t:zero()
  for i = #s, math.max(#s - self.length + 1, 1), -1 do
    if self.dict[s:sub(i,i)] then
      t[self.dict[s:sub(i,i)]][#s - i + 1] = 1
    end
  end
  return t
end












