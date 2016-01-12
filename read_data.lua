function read_embedding(vocab_path, emb_path)
  local vocab = Vocab(vocab_path)
  local embedding = torch.load(emb_path)
  return vocab, embedding
end

function read_sentences(path, vocab)
  local sentences = {}
  local file = io.open(path, 'r')
  local line
  while true do
    line = file:read()
    if line == nil then break end
    local tokens = stringx.split(line)
    local len = #tokens
    local sent = torch.IntTensor(len)
    for i = 1, len do
      local token = tokens[i]
      sent[i] = vocab:index(token)
    end
    sentences[#sentences + 1] = sent
  end

  file:close()
  return sentences
end

function read_dataset(dir, vocab)

  local entlabmap = {NEUTRAL=3, CONTRADICTION=1, ENTAILMENT=2}
  local dataset = {}
  dataset.vocab = vocab

  dataset.lsents = read_sentences(dir .. 'a.toks', vocab)
  dataset.rsents = read_sentences(dir .. 'b.toks', vocab)
  dataset.size = #dataset.lsents


  local sim_file = torch.DiskFile(dir .. 'sim.txt', 'r')
  local ent_file = io.open(dir .. 'ent.txt', 'r')
  dataset.sim_labels = torch.Tensor(dataset.size)
  dataset.ent_labels = torch.Tensor(dataset.size)


  for i = 1, dataset.size do
    dataset.sim_labels[i] = 0.25 * (sim_file:readDouble() - 1)
    dataset.ent_labels[i] = entlabmap[ent_file:read()]
  end

  sim_file:close()
  ent_file:close()

  return dataset
end