"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time

class RadixNode:
    def __init__(self, token=None):
        self.token = token
        self.children = {}
        self.pos = 0 
        self.kv_cache = None
    
class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def insert(self, tokens):
        cur = self.root
        for token in tokens:
            if token not in cur.children:
                cur.children[token] = RadixNode(token)
                cur.children[token].pos = cur.pos + 1
            cur = cur.children[token]
    
    def search(self, tokens):
        cur = self.root
        for i, token in enumerate(tokens):
            if token not in cur.children:
                return cur.kv_cache, tokens[i:] # return the last cache and the remaining tokens
            cur = cur.children[token]
        return cur.kv_cache, [] # exact match, return cache and empty list

# -----------------------------------------------------------------------------
init_from = 'gpt2-medium' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = "Django is framework in "
prompts = [
    "Django is a framework in python",
    "Django is a python framework used for",
    "Django is a ",
    "Django supports",
    "Python is a"
    ]

# num_samples = 10 # number of samples to draw
num_samples = 1
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
device = 'mps' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = 'float32'
compile = False # use PyTorch 2.0 to compile the model to be faster
use_cache = True # whether to use the past key/values cache during generation
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# RADIX TREE Changes ##########################################################################################################
# encode the prompts
encoded_prompts = [encode(p) for p in prompts]
max_prompt_length = max(len(p) for p in encoded_prompts)

tree = RadixTree()
for enc_prompts in encoded_prompts:
    tree.insert(enc_prompts)

# Set a 0 matrix of size (num_prompts, max_prompt_length) to hold the encoded prompts, and fill it with the encoded prompts
x = torch.zeros(len(encoded_prompts), max_prompt_length, dtype=torch.long, device=device) # size (num_prompts, max_prompt_length) => (B, max T)
for i, enc_prompts in enumerate(encoded_prompts):
    x[i, :len(enc_prompts)] = torch.tensor(enc_prompts, dtype=torch.long, device=device)

def build_kv_cache(model, node, device=device):
    """
    Recursively builds the KV cache for the Radix Tree.
    
    Strategy:
    1. We are currently at 'node'.
    2. We iterate through every child of 'node'.
    3. For each child:
       a. LOAD: Reset model to 'node's cache state.
       b. FORWARD: Pass the child's token through the model.
       c. SAVE: Snapshot the resulting cache into the child node.
       d. RECURSE: Go deeper.
   """
   # Iterate over all children of the current node
    for token, child_node in node.children.items():
        
        # --- STEP 1: RESTORE PARENT STATE ---
        # Before processing a child, we must ensure the model's cache 
        # is exactly what it was at the current node (the parent).
        if node.token is None: 
            # If we are at the Root, the cache should be empty.
            model.set_kv_caching(True)
        else:
            # Load the stored cache from the current node into the model layers
            for block, (k, v) in zip(model.transformer.h, node.kv_cache):
                block.attn.cache_k = k.clone() 
                block.attn.cache_v = v.clone()
                block.attn.is_caching_enabled = True

        # --- STEP 2: RUN FORWARD PASS ---
        # Run only the child's token. The model will append this 
        # to the restored cache.
        x_input = torch.tensor([[child_node.token]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            # We don't care about logits here, only the side-effect (cache update)
            model(x_input)

        # --- STEP 3: SNAPSHOT (SAVE) STATE ---
        # The model now contains [Parent Context] + [Child Token].
        # We save this state into the child node.
        child_node.kv_cache = []
        for block in model.transformer.h:
            # CRITICAL: .clone() is required. 
            # Otherwise, you save a reference that changes in the next iteration.
            k_cache = block.attn.cache_k.clone()
            v_cache = block.attn.cache_v.clone()
            child_node.kv_cache.append((k_cache, v_cache))

        # --- STEP 4: RECURSE ---
        build_kv_cache(model, child_node, device)

print("Building Radix Tree Cache...")
build_kv_cache(model, tree.root)
print("Cache build complete.")

# -----------------------------------------------------------------------------------------------------------------------------

def load_kv_cache_to_model(model, kv_cache):
    """ Helper to load the specific cache list into the model """
    if kv_cache is None:
        # If no cache found (root), clear it and ensure caching is on
        model.clear_kv_cache()
        model.set_kv_caching(True)
    else:
        # Load the stored cache
        for block, (k, v) in zip(model.transformer.h, kv_cache):
            block.attn.cache_k = k.clone()
            block.attn.cache_v = v.clone()
            block.attn.is_caching_enabled = True

print(f"Generating {num_samples} samples per prompt using Radix Tree...")

# run generation
with torch.no_grad():
    for prompt_idx, tokens in enumerate(encoded_prompts):
        
        # 1. SEARCH STRATEGY:
        # We search for the prefix (tokens[:-1]) instead of the full prompt.
        # This ensures we stop ONE step early. We will feed the last token 
        # into .generate() so the model can compute logits for it and sample the next token.
        if len(tokens) > 0:
            search_tokens = tokens[:-1]
            last_token = tokens[-1]
        else:
            continue

        # Search the tree
        # cache_found: The KV cache up to the match
        # suffix_found: Tokens in the search_list that weren't in the tree (the gap)
        cache_found, suffix_found = tree.search(search_tokens)

        # 2. PREPARE INPUT
        # The input to the model is: [Suffix not in tree] + [The Last Token]
        # Example: Prompt "A B C D". Tree has "A B".
        # Search "A B C" -> Finds "A B", returns Suffix "C".
        # Input to generate -> "C" + "D" = "C D".
        # Model loads "A B" cache, processes "C", then "D", then predicts "E".
        input_tokens = suffix_found + [last_token] if last_token is not None else []
        
        # Convert to tensor
        x_input = torch.tensor([input_tokens], dtype=torch.long, device=device)

        print(f"\nPrompt {prompt_idx}: {prompts[prompt_idx].strip()}")
        print(f"  - Prefix found length: {len(tokens) - len(input_tokens)}")
        print(f"  - Processing new tokens: {len(input_tokens)}")

        # 3. GENERATION
        with ctx:
            for k in range(num_samples):
                start = time.time()
                # A. Load the cache state for this specific prompt
                load_kv_cache_to_model(model, cache_found)

                # B. Generate
                # Note: We subtract len(input_tokens) from max_new_tokens if you want exact length control,
                # but standard practice is usually max_new_tokens *added*.
                # We use x_input to "kickstart" the generation.
                y = model.generate(x_input, max_new_tokens, temperature=temperature, top_k=top_k, use_cache=use_cache)
                
                # C. Reconstruct full text for display
                # We need to combine the prefix (from tree) + the new generation (y)
                # The 'y' tensor contains [input_tokens + generated_tokens]
                prefix_len = len(tokens) - len(input_tokens)
                full_sequence = tokens[:prefix_len] + y[0].tolist()
                
                print(f"  Sample {k}: {decode(full_sequence)}")
                print('  ---------------')
                end = time.time()
                print(f"Time required to run {end-start} sec.")
                print('  ---------------')
