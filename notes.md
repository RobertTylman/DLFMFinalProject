
⏺ Full track on the DB side; 10 s only on the query side. That's how it should work — and matches how real Shazam works.                       

- build_gtzan_db.py calls index_folder() (src/index_directory.py:32-60), which runs process_audio_pipeline(filepath) on the whole file (~30 s for GTZAN), extracts peaks from the full spectrogram, and stores every hash → (song_id, offset) row in fingerprints_gtzan.db. So a      
  single GTZAN track contributes thousands of hashes spanning the whole 30 s.                                                                                                                                                                                                           
  - evaluate_shazam.py is the only place the 10 s snippet exists. It writes the snippet to a temp WAV, fingerprints it, and queries the DB. The matcher's time-coherence scoring (identify.py:111-118) computes diff = db_offset - snippet_offset for every matched hash and finds the  
  dominant (song_id, diff) bucket — that diff recovers where in the 30 s original the snippet came from.                                                                                                                                                                                
                                                                                                                                                                                                                                                                                        
  This asymmetry is the whole point of the design: indexing is one-time and exhaustive, queries are short and cheap. Indexing only 10 s of each original would discard 2/3 of the reference hashes and break exactly the scenario Shazam is built for (snippet from anywhere in the     
  song).                                                    
                                                                                                                                                           

 python3 Shazam/evaluation/build_gtzan_db.py            # ~minutes, one time
 python3 Shazam/evaluation/evaluate_shazam.py --limit 5  # smoke test
 python3 Shazam/evaluation/evaluate_shazam.py            # full ~8,991 files

Why it's still a defensible experiment: your stated goal in the README is comparative — "Shazam-style fingerprinting vs. CLAP embeddings under degradation." As long as both methods are evaluated against the same augmented files, the bias applies equally to both, and relative   
  claims ("Method A degrades faster than Method B between 10 dB and 0 dB") remain valid. The bias only invalidates absolute claims like "Shazam achieves X% in noisy environments."
                                                                                                                                                                                                                                                                                        
  Cheap mitigations if you want to narrow the gap (in rough order of bang-for-buck):                                                                                                                                                                                                    
  - Add an MP3/AAC re-encode pass to the snippet (codec artifacts hit fingerprinting hard) — ffmpeg -i in.wav -b:a 64k out.aac && ffmpeg -i out.aac out.wav
  - Convolve with a short room IR before mixing noise (one extra scipy.signal.fftconvolve call)                                                                                                                                                                                         
  - Apply a phone-mic-ish bandpass (e.g. 300 Hz–3.4 kHz lowpass) — your framework table already lists lowpass as a planned transform
                                                                                                                                                                                                                                                                                        
  I'd flag the bias explicitly in your final write-up but not change the pipeline unless you want to make absolute-accuracy claims. Want me to add a "Limitations / Bias" subsection under the Shazam Evaluation section in the main README?             
