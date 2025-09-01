import streamlit as st
import time

message = st.chat_message(name="Yo Mama", avatar="👿", width="stretch")

message.write("Waddup Bitch ass mfer!!!")
time.sleep(4.0)              
message.write("👿 Welcome to my App where I like to talk my shit!"
" And you aint gone do shit about it!!!") 
time.sleep(6.0)
message.write(" 👿 I heard you like diddy parties. You probably got diddied with ya bitch ass...")
time.sleep(6.0)
st.write('👿 I just wanted yall all to know ERIC IS A BITCH!!! AND DONT YALL FORGET IT!!')
time.sleep(3.0)
st.title(' 👿 Beep Bop Boop! Now Calculating if you\'re a bitch...')

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Calculating... {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

st.title('🚨 🚨 🚨 Bitch Alert!!! 🚨 🚨 🚨 Bitch Alert!!🚨 🚨 🚨')

'BEEEEEEEEEEEEP!!!! ...Confirmed, Bitch. Initializing Diddy party'











