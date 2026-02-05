
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='input.txt', 
    model_prefix='bpe_model',  
    vocab_size=8000,  
    character_coverage=0.995, 
    model_type='bpe'  
)
