import os
import torch
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1
import re

class ProstT5EmbeddingGenerator:
    def __init__(self, model_name="Rostlab/ProstT5", device='cuda'):
        """Initialize ProstT5 model"""
        print(f"Loading ProstT5 model on {device}...")
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        
        # Use safetensors to avoid the torch.load vulnerability issue
        self.model = T5EncoderModel.from_pretrained(
            model_name, 
            use_safetensors=True  # This will use the .safetensors file instead
        ).to(device)
        
        self.model.eval()
        print("✓ Model loaded successfully!")
    
    def extract_sequence_from_pdb(self, pdb_file):
        """Extract amino acid sequence from PDB file"""
        parser = PDBParser(QUIET=True)
        
        try:
            structure = parser.get_structure('protein', pdb_file)
            
            sequences = []
            for model in structure:
                for chain in model:
                    seq = []
                    for residue in chain:
                        if residue.id[0] == ' ':  # Standard residue
                            resname = residue.get_resname()
                            if resname in protein_letters_3to1:
                                seq.append(protein_letters_3to1[resname])
                    if seq:
                        sequences.append(''.join(seq))
            
            # Take the longest chain if multiple
            if sequences:
                sequence = max(sequences, key=len)
                return sequence
            else:
                return None
                
        except Exception as e:
            print(f"  Error parsing PDB: {e}")
            return None
    
    def load_3di_sequence(self, three_di_file):
        """Load 3Di sequence from file"""
        try:
            with open(three_di_file, 'r') as f:
                lines = f.readlines()
                # Skip header line (starts with >)
                sequence = ''.join([line.strip() for line in lines if not line.startswith('>')])
                return sequence
        except Exception as e:
            print(f"  Error loading 3Di file: {e}")
            return None
    
    def generate_embedding(self, sequence_3di, sequence_aa):
        """
        Generate ProstT5 embedding using proper dual-prefix approach.
        Returns mean-pooled (1024,) or per-residue (L, 2048) embedding.
        """
        import re
        
        L = min(len(sequence_aa), len(sequence_3di))
        sequence_aa  = sequence_aa[:L]
        sequence_3di = sequence_3di[:L]

        seq_aa_spaced  = " ".join(list(re.sub(r"[UZOB]", "X", sequence_aa)))
        seq_3di_spaced = " ".join(list(sequence_3di.lower()))

        input_seqs = [
            "<AA2fold> "  + seq_aa_spaced,   # AA track
            "<fold2AA> "  + seq_3di_spaced,  # 3Di track
        ]

        ids = self.tokenizer(
            input_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(ids.input_ids, attention_mask=ids.attention_mask)

        # Skip prefix token [0], take L residue tokens
        emb_aa  = outputs.last_hidden_state[0, 1 : L + 1]  # (L, 1024)
        emb_3di = outputs.last_hidden_state[1, 1 : L + 1]  # (L, 1024)

        # Concatenate both tracks → (L, 2048)
        combined = torch.cat([emb_aa, emb_3di], dim=-1)

        return combined.cpu().numpy()  # (L, 2048)
    
    def process_all_proteins(self, pdb_dir, three_di_dir, output_dir):
        """
        Process all proteins and generate embeddings
        
        Args:
            pdb_dir: Directory with PDB files
            three_di_dir: Directory with .3di files
            output_dir: Where to save embeddings
        """
        pdb_dir = Path(pdb_dir)
        three_di_dir = Path(three_di_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all PDB files
        pdb_files = sorted(list(pdb_dir.glob("*.pdb")))
        print(f"\nFound {len(pdb_files)} PDB files")
        
        successful = 0
        failed = 0
        failed_files = []
        
        for i, pdb_file in enumerate(pdb_files, 1):
            protein_name = pdb_file.stem
            
            print(f"\n[{i}/{len(pdb_files)}] Processing: {protein_name}")
            
            # Find corresponding 3Di file
            three_di_file = three_di_dir / f"{protein_name}.3di"
            
            if not three_di_file.exists():
                print(f"  ✗ 3Di file not found: {three_di_file}")
                failed += 1
                failed_files.append((protein_name, "Missing 3Di file"))
                continue
            
            try:
                # Extract sequences
                print(f"  Extracting amino acid sequence from PDB...")
                sequence_aa = self.extract_sequence_from_pdb(pdb_file)
                
                if sequence_aa is None:
                    print(f"  ✗ Failed to extract sequence from PDB")
                    failed += 1
                    failed_files.append((protein_name, "Failed to extract AA sequence"))
                    continue
                
                print(f"    AA sequence length: {len(sequence_aa)}")
                
                print(f"  Loading 3Di sequence...")
                sequence_3di = self.load_3di_sequence(three_di_file)
                
                if sequence_3di is None:
                    print(f"  ✗ Failed to load 3Di sequence")
                    failed += 1
                    failed_files.append((protein_name, "Failed to load 3Di"))
                    continue
                
                print(f"    3Di sequence length: {len(sequence_3di)}")
                
                # Check length match (should be similar, allow some tolerance)
                if abs(len(sequence_aa) - len(sequence_3di)) > 5:
                    print(f"  ⚠ Warning: Length mismatch - AA:{len(sequence_aa)} vs 3Di:{len(sequence_3di)}")
                    # Use the shorter length
                    min_len = min(len(sequence_aa), len(sequence_3di))
                    sequence_aa = sequence_aa[:min_len]
                    sequence_3di = sequence_3di[:min_len]
                    print(f"    Truncated both to length: {min_len}")
                
                # Generate embedding
                print(f"  Generating ProstT5 embedding...")
                embedding = self.generate_embedding(sequence_3di, sequence_aa)
                
                print(f"    Embedding shape: {embedding.shape}")
                
                # Save embedding as flat text file (compatible with your loader)
                output_file = output_dir / f"{protein_name}.fea"
                embedding_flat = embedding.flatten()
                np.savetxt(output_file, embedding_flat)
                
                print(f"  ✓ Saved to: {output_file}")
                print(f"    Total elements: {len(embedding_flat)} ({embedding.shape[0]} residues × 2048)")
                
                successful += 1
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                failed_files.append((protein_name, str(e)))
                continue
        
        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successful: {successful}/{len(pdb_files)}")
        print(f"✗ Failed: {failed}/{len(pdb_files)}")
        
        if failed_files:
            print(f"\nFailed files:")
            for name, reason in failed_files:
                print(f"  - {name}: {reason}")
        
        print(f"\n✅ All embeddings saved to: {output_dir}")


def main():
    HOME_DIR = "/path/to/HOME_DIR"

    PDB_DIR = os.path.join(
        HOME_DIR,
        "Protein_data/AF_structure"
    )

    THREE_DI_DIR = os.path.join(
        HOME_DIR,
        "Protein_data/3Di_tokens"
    )

    OUTPUT_DIR = os.path.join(
        HOME_DIR,
        "Protein_data/prostt5_featuresV1"
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    generator = ProstT5EmbeddingGenerator(device=device)

    generator.process_all_proteins(
        pdb_dir=PDB_DIR,
        three_di_dir=THREE_DI_DIR,
        output_dir=OUTPUT_DIR
    )

    print("\nDone!")


if __name__ == "__main__":
    main()