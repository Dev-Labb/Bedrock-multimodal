import streamlit as st
import streamlit as st
import time

message = st.chat_message(name="Yo Mama", avatar="👿", width="stretch")

message.write("Waddup Bitch ass mfer!!!" \
"Welcome to my App where I like to talk my shit!" \
" And you aint gone do shit about it!!! " \
"I heard you like diddy parties." \
"You probably got diddied ya with ya bitch ass...")

st.write('👿 I just wanted yall all to know ERIC IS A BITCH!!! AND DONT YALL FORGET IT!!')

st.title('Beep Bop Boop Calculating if you\'re a bitch...')

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Calculating... {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'BEEEEP...Confirmed. You are indeed a lil bitch'
