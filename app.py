import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from PIL import Image
import time
import math

# Set page configuration
st.set_page_config(page_title="Word Embedding Visualization", layout="wide")

# Title and description
st.title("Word Embedding Visualization in 3D Space")
st.markdown("""
Aplikasi ini mendemonstrasikan bagaimana word embeddings berkembang selama pelatihan dalam Natural Language Processing (NLP).
Lihat bagaimana kata-kata bergerak dalam ruang 3D, dengan kata-kata yang memiliki makna serupa mengelompok bersama seiring waktu.
""")

# Sidebar for controls
st.sidebar.header("Kontrol")

# Function to generate sample word embeddings
def generate_word_embeddings(vocab_size=100, embedding_dim=100, num_epochs=100):
    # Sample vocabulary (most common English words)
    vocabulary = [
        # Basic pronouns and articles
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        # Additional common words
        "time", "know", "people", "year", "good", "some", "now", "very", "just", "like",
        "other", "then", "come", "these", "could", "here", "first", "than", "been", "call",
        "way", "work", "new", "make", "over", "think", "take", "any", "day", "most",
        "us", "him", "them", "man", "woman", "child", "our", "your", "can", "may",
        "should", "must", "such", "great", "still", "after", "before", "through", "same", "between"
    ]
    
    # Create semantic relationships between certain words
    related_words = {
        "he": ["his", "him", "man"],
        "she": ["her", "woman"],
        "I": ["me", "my"],
        "they": ["their", "them"],
        "we": ["us", "our"],
        "you": ["your"],
        "child": ["man", "woman"],
        "time": ["day", "year"],
        "good": ["great"],
        "should": ["must", "may"],
        "before": ["after"],
    }
    
    # Initialize random embeddings
    np.random.seed(42)  # For reproducibility
    initial_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
    
    # Create embeddings for each epoch
    all_epoch_embeddings = []
    
    for epoch in range(num_epochs + 1):
        if epoch == 0:
            # Initial random embeddings
            all_epoch_embeddings.append(initial_embeddings.copy())
        else:
            # Get previous epoch's embeddings
            prev_embeddings = all_epoch_embeddings[-1].copy()
            
            # Update embeddings to simulate training
            # Add small random changes
            random_changes = np.random.randn(vocab_size, embedding_dim) * 0.01
            
            # Add directed changes for related words (make them move closer)
            for key_word, related in related_words.items():
                if key_word in vocabulary:
                    key_idx = vocabulary.index(key_word)
                    key_embedding = prev_embeddings[key_idx]
                    
                    for rel_word in related:
                        if rel_word in vocabulary:
                            rel_idx = vocabulary.index(rel_word)
                            # Move related word embedding closer to key word
                            direction = key_embedding - prev_embeddings[rel_idx]
                            # The closer we get to the final epoch, the closer related words should be
                            strength = 0.05 * (epoch / num_epochs)
                            prev_embeddings[rel_idx] += direction * strength
            
            # Add the random noise
            new_embeddings = prev_embeddings + random_changes
            all_epoch_embeddings.append(new_embeddings)
    
    return vocabulary, all_epoch_embeddings

# Function to reduce dimensions for visualization
def reduce_dimensions(embeddings, method='tsne', dimensions=3):
    if method == 'tsne':
        tsne = TSNE(n_components=dimensions, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced_embeddings = tsne.fit_transform(embeddings)
    else:
        # Use first 3 dimensions (simple but less effective)
        reduced_embeddings = embeddings[:, :dimensions]
    
    return reduced_embeddings

# Function to create a 3D scatter plot
def create_3d_plot(words, embeddings, epoch, highlight_words=None):
    # Reduce dimensions to 3D
    embeddings_3d = reduce_dimensions(embeddings)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'word': words,
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        text='word',
        title=f'Word Embeddings at Epoch {epoch}',
        opacity=0.7,
    )
    
    # Update marker size and text position
    fig.update_traces(
        marker=dict(size=5),
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}'
    )
    
    # Highlight specific words if requested
    if highlight_words:
        highlight_indices = [words.index(word) for word in highlight_words if word in words]
        highlight_df = df.iloc[highlight_indices]
        
        fig.add_trace(go.Scatter3d(
            x=highlight_df['x'],
            y=highlight_df['y'],
            z=highlight_df['z'],
            text=highlight_df['word'],
            mode='markers+text',
            marker=dict(size=8, color='red'),
            textposition='top center',
            name='Highlighted Words'
        ))
    
    # Update layout for better visualization and interactivity
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            # Add camera controls for better interactivity
            dragmode='turntable',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    # Add buttons for camera control
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Reset View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=1.25, y=1.25, z=1.25))]
                    ),
                    dict(
                        label='Top View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=0, y=0, z=2.5))]
                    ),
                    dict(
                        label='Side View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=2.5, y=0, z=0))]
                    )
                ],
                direction='down',
                pad={'r': 10, 't': 10},
                x=0.1,
                y=1.1,
                xanchor='left',
                yanchor='top'
            )
        ]
    )
    
    # Add annotation to explain how to interact with the plot
    fig.add_annotation(
        text="Klik dan seret untuk memutar. Gunakan scroll untuk zoom. Klik dua kali untuk reset.",
        xref="paper", yref="paper",
        x=0.5, y=0,
        showarrow=False,
        font=dict(size=10)
    )
    
    return fig

# Function to create an animation of embeddings evolution
def create_embedding_animation(words, all_embeddings, selected_words=None, interval=10):
    # Select epochs at regular intervals
    num_epochs = len(all_embeddings) - 1
    selected_epochs = list(range(0, num_epochs + 1, interval))
    if num_epochs not in selected_epochs:
        selected_epochs.append(num_epochs)
    
    # Create a figure for matplotlib animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Function to update the plot for each frame
    def update(frame_idx):
        epoch = selected_epochs[frame_idx]
        ax.clear()
        
        # Get embeddings for this epoch and reduce to 3D
        embeddings = all_embeddings[epoch]
        embeddings_3d = reduce_dimensions(embeddings)
        
        # Plot all words
        ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], alpha=0.6)
        
        # Add text labels for all words
        for i, word in enumerate(words):
            ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], word, size=8)
        
        # Highlight selected words if specified
        if selected_words:
            highlight_indices = [words.index(word) for word in selected_words if word in words]
            highlight_points = embeddings_3d[highlight_indices]
            ax.scatter(highlight_points[:, 0], highlight_points[:, 1], highlight_points[:, 2], 
                      color='red', s=100, alpha=0.8)
        
        # Add special title for final epoch
        if epoch == num_epochs:
            ax.set_title(f'Word Embeddings at Epoch {epoch} (Hasil Akhir)')
        else:
            ax.set_title(f'Word Embeddings at Epoch {epoch}')
        
        ax.set_axis_off()
    
    # Create animation with slower speed (lower fps) and extra frames for the last epoch
    # Create frame sequence with duplicated final frame to create pause effect
    frames = list(range(len(selected_epochs)))
    
    # Duplicate the last frame multiple times to create a pause at the end
    last_frame = frames[-1]
    pause_frames = [last_frame] * 10  # Add 10 copies of the last frame
    all_frames = frames + pause_frames
    
    # Create animation with slower speed (lower fps)
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=800)  # Slower animation (800ms between frames)
    
    # Create a temporary file to save the animation
    import tempfile
    import os
    
    # Create a temporary file with .gif extension
    temp_file = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    try:
        # Save animation to the temporary file with slower fps
        anim.save(temp_filename, writer='pillow', fps=1)  # Slower fps for better viewing
        
        # Read the file into a buffer for Streamlit
        with open(temp_filename, 'rb') as f:
            gif_data = f.read()
            
        # Create a BytesIO object from the data
        gif_buffer = io.BytesIO(gif_data)
        gif_buffer.seek(0)
        
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
    
    plt.close(fig)  # Close the figure to free memory
    
    return gif_buffer

# Main application logic
# Function to generate positional encoding
def generate_positional_encoding(max_seq_len=100, d_model=512):
    # Create positional encoding matrix
    pos_encoding = np.zeros((max_seq_len, d_model))
    
    # Calculate positional encodings
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return pos_encoding

# Function to visualize positional encoding
def visualize_positional_encoding(pos_encoding, max_positions=20, max_dims=64):
    # Limit the visualization to a subset of positions and dimensions
    subset_encoding = pos_encoding[:max_positions, :max_dims]
    
    # Create heatmap
    fig = px.imshow(
        subset_encoding,
        labels=dict(x="Dimensi", y="Posisi", color="Nilai"),
        x=[f"Dim {i+1}" for i in range(max_dims)],
        y=[f"Pos {i+1}" for i in range(max_positions)],
        color_continuous_scale="RdBu_r",
        title="Visualisasi Positional Encoding"
    )
    
    # Update layout
    fig.update_layout(
        xaxis=dict(side="top"),
        height=500
    )
    
    return fig

# Function to visualize sinusoidal patterns
def visualize_sinusoidal_patterns(pos_encoding, positions=[0, 10, 20, 30], max_dims=64):
    # Create figure
    fig = go.Figure()
    
    # Add traces for each position
    for pos in positions:
        if pos < len(pos_encoding):
            fig.add_trace(go.Scatter(
                x=list(range(max_dims)),
                y=pos_encoding[pos, :max_dims],
                mode='lines',
                name=f'Posisi {pos+1}'
            ))
    
    # Update layout
    fig.update_layout(
        title="Pola Sinusoidal Positional Encoding untuk Posisi yang Berbeda",
        xaxis_title="Dimensi",
        yaxis_title="Nilai",
        legend_title="Posisi",
        height=400
    )
    
    return fig

# Function to combine word embeddings with positional encoding
def combine_embeddings_with_position(embeddings, pos_encoding, positions, d_model=None):
    # If d_model is not provided, use the embedding dimension
    if d_model is None:
        d_model = embeddings.shape[1]
    
    # Ensure positional encoding has the right dimensions
    if pos_encoding.shape[1] != d_model:
        # Resize positional encoding if needed
        resized_pos_encoding = np.zeros((pos_encoding.shape[0], d_model))
        min_dim = min(pos_encoding.shape[1], d_model)
        resized_pos_encoding[:, :min_dim] = pos_encoding[:, :min_dim]
        pos_encoding = resized_pos_encoding
    
    # Create combined embeddings
    combined_embeddings = []
    for i, position in enumerate(positions):
        if position < pos_encoding.shape[0]:
            # Get positional encoding for this position
            pos_vector = pos_encoding[position]
            
            # Add positional encoding to each word embedding
            combined = embeddings + pos_vector
            combined_embeddings.append(combined)
    
    return combined_embeddings

# Function to visualize the combination process
def visualize_embedding_combination(word, embedding, pos_encoding, position, max_dims=20):
    # Ensure we only show a subset of dimensions for clarity
    embedding_subset = embedding[:max_dims]
    pos_encoding_subset = pos_encoding[position, :max_dims]
    combined_subset = embedding_subset + pos_encoding_subset
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Dimensi': list(range(1, max_dims + 1)),
        'Word Embedding': embedding_subset,
        'Positional Encoding': pos_encoding_subset,
        'Combined Representation': combined_subset
    })
    
    # Melt the DataFrame for easier plotting
    df_melted = pd.melt(df, id_vars=['Dimensi'], 
                        value_vars=['Word Embedding', 'Positional Encoding', 'Combined Representation'],
                        var_name='Komponen', value_name='Nilai')
    
    # Create the plot
    fig = px.line(df_melted, x='Dimensi', y='Nilai', color='Komponen', markers=True,
                 title=f'Kombinasi Word Embedding dan Positional Encoding untuk "{word}" pada Posisi {position+1}')
    
    # Update layout
    fig.update_layout(
        xaxis_title="Dimensi",
        yaxis_title="Nilai",
        legend_title="Komponen",
        height=400
    )
    
    return fig

# Function to create 3D visualization of combined embeddings
def visualize_combined_embeddings_3d(words, embeddings, pos_encoding, positions, highlight_words=None):
    # Combine embeddings with positional encoding
    combined_embeddings = combine_embeddings_with_position(embeddings, pos_encoding, positions)
    
    # Create a list to store all points for the 3D plot
    all_points = []
    all_labels = []
    all_colors = []
    all_sizes = []
    all_symbols = []
    
    # Add original embeddings
    embeddings_3d = reduce_dimensions(embeddings)
    for i, word in enumerate(words):
        all_points.append(embeddings_3d[i])
        all_labels.append(f"{word} (original)")
        all_colors.append('blue')
        all_sizes.append(5)
        all_symbols.append('circle')
    
    # Add combined embeddings for each position
    for p_idx, position in enumerate(positions):
        if p_idx < len(combined_embeddings):
            combined_3d = reduce_dimensions(combined_embeddings[p_idx])
            for i, word in enumerate(words):
                all_points.append(combined_3d[i])
                all_labels.append(f"{word} (pos {position+1})")
                all_colors.append(px.colors.qualitative.Plotly[p_idx % len(px.colors.qualitative.Plotly)])
                all_sizes.append(5)
                all_symbols.append('diamond')
    
    # Convert to numpy array for easier manipulation
    all_points = np.array(all_points)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'label': all_labels,
        'x': all_points[:, 0],
        'y': all_points[:, 1],
        'z': all_points[:, 2],
        'color': all_colors,
        'size': all_sizes,
        'symbol': all_symbols
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        text='label',
        color='color',
        size='size',
        title=f'Word Embeddings with Positional Encoding',
        opacity=0.7,
    )
    
    # Update marker size and text position
    fig.update_traces(
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}'
    )
    
    # Highlight specific words if requested
    if highlight_words:
        # Highlight in original embeddings
        highlight_indices = [words.index(word) for word in highlight_words if word in words]
        
        for p_idx, position in enumerate(positions):
            if p_idx < len(combined_embeddings):
                # Calculate offset for this position's embeddings
                offset = len(words) * (p_idx + 1)
                
                # Add highlighted points for this position
                highlight_points = [offset + idx for idx in highlight_indices]
                highlight_df = df.iloc[highlight_points]
                
                fig.add_trace(go.Scatter3d(
                    x=highlight_df['x'],
                    y=highlight_df['y'],
                    z=highlight_df['z'],
                    text=highlight_df['label'],
                    mode='markers+text',
                    marker=dict(size=8, color='red'),
                    textposition='top center',
                    name=f'Highlighted Words (Pos {position+1})'
                ))
    
    # Update layout for better visualization and interactivity
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
            dragmode='turntable',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    # Add buttons for camera control
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Reset View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=1.25, y=1.25, z=1.25))]
                    ),
                    dict(
                        label='Top View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=0, y=0, z=2.5))]
                    ),
                    dict(
                        label='Side View',
                        method='relayout',
                        args=['scene.camera', dict(eye=dict(x=2.5, y=0, z=0))]
                    )
                ],
                direction='down',
                pad={'r': 10, 't': 10},
                x=0.1,
                y=1.1,
                xanchor='left',
                yanchor='top'
            )
        ]
    )
    
    # Add annotation to explain how to interact with the plot
    fig.add_annotation(
        text="Klik dan seret untuk memutar. Gunakan scroll untuk zoom. Klik dua kali untuk reset.",
        xref="paper", yref="paper",
        x=0.5, y=0,
        showarrow=False,
        font=dict(size=10)
    )
    
    return fig

def main():
    # Sidebar controls
    st.sidebar.header("Parameter")
    vocab_size = st.sidebar.slider("Ukuran Vocabulary", 10, 100, 100)
    embedding_dim = st.sidebar.slider("Dimensi Embedding", 50, 300, 100)
    num_epochs = st.sidebar.slider("Jumlah Epoch Pelatihan", 50, 200, 100)
    update_interval = st.sidebar.slider("Interval Update (epoch)", 5, 20, 10)
    
    # Generate word embeddings
    with st.spinner("Menghasilkan word embeddings..."):
        vocabulary, all_epoch_embeddings = generate_word_embeddings(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim, 
            num_epochs=num_epochs
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Word Embedding", "Positional Encoding", "Combined Representation"])
    
    with tab1:
        # Word Embedding Visualization Section
        with st.expander("Lihat Vocabulary", expanded=False):
            st.write(vocabulary)
        
        # Word selection for highlighting
        with st.expander("Pilih Kata untuk Disorot", expanded=True):
            selected_words = st.multiselect("Pilih kata untuk dilacak", vocabulary, 
                                           default=["he", "she", "his", "her"])
        
        # Display current epoch embeddings
        with st.expander("Visualisasi Epoch Saat Ini", expanded=True):
            # Epoch selection
            selected_epoch = st.select_slider(
                "Pilih Epoch", 
                options=list(range(0, num_epochs + 1)),
                value=0
            )
            
            # Display 3D visualization for selected epoch
            fig = create_3d_plot(
                vocabulary, 
                all_epoch_embeddings[selected_epoch], 
                selected_epoch,
                highlight_words=selected_words
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display embedding values for selected words
        if selected_words:
            with st.expander("Nilai Embedding untuk Kata yang Dipilih", expanded=False):
                # Get indices of selected words
                selected_indices = [vocabulary.index(word) for word in selected_words if word in vocabulary]
                
                # Create DataFrame with embedding values
                embedding_values = all_epoch_embeddings[selected_epoch][selected_indices]
                
                # Display first few dimensions of embeddings
                display_dims = min(5, embedding_dim)  # Show at most 5 dimensions
                df_display = pd.DataFrame(
                    embedding_values[:, :display_dims], 
                    index=selected_words,
                    columns=[f"Dim {i+1}" for i in range(display_dims)]
                )
                
                st.dataframe(df_display)
                
                if display_dims < embedding_dim:
                    st.info(f"Menampilkan hanya {display_dims} dimensi pertama dari total {embedding_dim}.")
        
        # Create and display animation or interactive visualization
        with st.expander("Visualisasi Evolusi Embedding", expanded=True):
            # Add option to choose between animation and interactive visualization
            viz_type = st.radio(
                "Pilih Jenis Visualisasi",
                ["Animasi GIF", "Visualisasi Interaktif 3D"],
                index=1  # Default to interactive visualization
            )
            
            if viz_type == "Animasi GIF":
                if st.button("Hasilkan Animasi"):
                    with st.spinner("Menghasilkan animasi... Ini mungkin memerlukan waktu sebentar."):
                        gif_buffer = create_embedding_animation(
                            vocabulary, 
                            all_epoch_embeddings, 
                            selected_words=selected_words,
                            interval=update_interval
                        )
                        
                        # Display the animation
                        st.image(gif_buffer, caption="Evolusi Word Embedding", use_column_width=True)
                        
                        # Provide download link for the GIF
                        gif_data = gif_buffer.getvalue()
                        b64 = base64.b64encode(gif_data).decode()
                        href = f'<a href="data:image/gif;base64,{b64}" download="word_embeddings.gif">Unduh GIF</a>'
                        st.markdown(href, unsafe_allow_html=True)
            else:  # Interactive 3D visualization
                # Add slider for epoch selection
                interactive_epoch = st.slider(
                    "Pilih Epoch untuk Visualisasi", 
                    min_value=0, 
                    max_value=num_epochs,
                    value=0,
                    step=max(1, num_epochs // 20)  # Create reasonable step size
                )
                
                # Add play button for automatic progression
                col1, col2 = st.columns([1, 3])
                with col1:
                    play_button = st.button("▶️ Putar Evolusi")
                with col2:
                    speed = st.select_slider(
                        "Kecepatan",
                        options=["Sangat Lambat", "Lambat", "Sedang", "Cepat"],
                        value="Sedang"
                    )
                
                # Create interactive 3D plot
                interactive_fig = create_3d_plot(
                    vocabulary, 
                    all_epoch_embeddings[interactive_epoch], 
                    interactive_epoch,
                    highlight_words=selected_words
                )
                
                # Display the interactive plot
                st.plotly_chart(interactive_fig, use_container_width=True)
                
                # Add instructions for interaction
                st.info(
                    "**Cara Berinteraksi dengan Visualisasi 3D:**\n"
                    "- Klik dan seret untuk memutar visualisasi\n"
                    "- Scroll untuk memperbesar/memperkecil\n"
                    "- Klik dua kali untuk mereset tampilan\n"
                    "- Gunakan slider epoch untuk melihat perubahan antar epoch\n"
                    "- Klik tombol 'Putar Evolusi' untuk melihat perubahan secara otomatis"
                )
                
                # Add JavaScript for auto-play functionality if button is clicked
                if play_button:
                    # Set delay based on selected speed
                    if speed == "Sangat Lambat":
                        delay = 2.0
                    elif speed == "Lambat":
                        delay = 1.5
                    elif speed == "Sedang":
                        delay = 1.0
                    else:  # Fast
                        delay = 0.5
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    # Auto-play through epochs
                    for i, epoch in enumerate(range(0, num_epochs + 1)):
                        # Update progress
                        progress = int(100 * i / num_epochs)
                        progress_bar.progress(progress)
                        
                        # Create and display plot for current epoch
                        play_fig = create_3d_plot(
                            vocabulary, 
                            all_epoch_embeddings[epoch], 
                            epoch,
                            highlight_words=selected_words
                        )
                        plot_container = st.empty()
                        plot_container.plotly_chart(play_fig, use_container_width=True)
                        
                        # Add extra delay for final epoch
                        if epoch == num_epochs:
                            time.sleep(delay * 3)  # Longer pause at the end
                            st.success("Visualisasi selesai! Ini adalah hasil akhir dari proses embedding.")
                        else:
                            time.sleep(delay)
                    
                    # Reset progress bar when done
                    progress_bar.progress(100)
    
    with tab2:
        # Positional Encoding Visualization Section
        st.header("Visualisasi Positional Encoding")
        st.markdown("""
        Positional encoding adalah komponen penting dalam model Transformer yang memungkinkan model untuk memahami urutan kata dalam kalimat.
        Transformer menggunakan fungsi sinusoidal untuk menghasilkan vektor posisi yang unik untuk setiap posisi dalam kalimat.
        """)
        
        # Parameters for positional encoding
        with st.expander("Parameter Positional Encoding", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                max_seq_len = st.slider("Panjang Sekuens Maksimum", 10, 200, 100)
            with col2:
                d_model = st.slider("Dimensi Model (d_model)", 64, 512, 256, step=64)
        
        # Generate positional encoding
        pos_encoding = generate_positional_encoding(max_seq_len, d_model)
        
        # Visualize positional encoding as heatmap
        with st.expander("Heatmap Positional Encoding", expanded=True):
            max_positions = st.slider("Jumlah Posisi untuk Ditampilkan", 5, 50, 20)
            max_dims = st.slider("Jumlah Dimensi untuk Ditampilkan", 16, 128, 64, step=16)
            
            heatmap_fig = visualize_positional_encoding(pos_encoding, max_positions, max_dims)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.markdown("""
            **Penjelasan Heatmap:**
            - Setiap baris mewakili posisi kata dalam kalimat (dari atas ke bawah)
            - Setiap kolom mewakili dimensi dalam vektor encoding
            - Warna menunjukkan nilai encoding (merah untuk positif, biru untuk negatif)
            """)
        
        # Visualize sinusoidal patterns
        with st.expander("Pola Sinusoidal", expanded=True):
            positions = st.multiselect(
                "Pilih posisi untuk divisualisasikan", 
                options=list(range(0, max_seq_len)),
                default=[0, 10, 20, 30]
            )
            
            if positions:
                sinusoidal_fig = visualize_sinusoidal_patterns(pos_encoding, positions, max_dims)
                st.plotly_chart(sinusoidal_fig, use_container_width=True)
                
                st.markdown("""
                **Penjelasan Pola Sinusoidal:**
                - Grafik menunjukkan bagaimana nilai encoding bervariasi di seluruh dimensi untuk posisi yang berbeda
                - Frekuensi sinusoidal menurun seiring dengan peningkatan dimensi
                - Posisi yang berbeda memiliki pola yang berbeda, memungkinkan model untuk membedakan posisi
                """)
        
        # Visualize position similarity
        with st.expander("Kesamaan Posisi", expanded=True):
            # Calculate similarity matrix
            similarity_matrix = np.zeros((max_positions, max_positions))
            for i in range(max_positions):
                for j in range(max_positions):
                    # Cosine similarity
                    dot_product = np.dot(pos_encoding[i, :max_dims], pos_encoding[j, :max_dims])
                    norm_i = np.linalg.norm(pos_encoding[i, :max_dims])
                    norm_j = np.linalg.norm(pos_encoding[j, :max_dims])
                    similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
            
            # Create heatmap
            similarity_fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Posisi", y="Posisi", color="Kesamaan"),
                x=[f"Pos {i+1}" for i in range(max_positions)],
                y=[f"Pos {i+1}" for i in range(max_positions)],
                color_continuous_scale="Viridis",
                title="Matriks Kesamaan Posisi (Cosine Similarity)"
            )
            
            st.plotly_chart(similarity_fig, use_container_width=True)
            
            st.markdown("""
            **Penjelasan Matriks Kesamaan:**
            - Warna terang menunjukkan posisi yang memiliki encoding yang serupa
            - Diagonal utama selalu memiliki kesamaan 1.0 (kesamaan sempurna dengan diri sendiri)
            - Posisi yang berdekatan cenderung memiliki kesamaan yang lebih tinggi
            - Pola ini memungkinkan model untuk memahami jarak relatif antara kata-kata
            """)
    
    with tab3:
        # Combined Representation Visualization Section
        st.header("Visualisasi Representasi Gabungan")
        st.markdown("""
        Pada model Transformer, representasi akhir dari sebuah kata adalah kombinasi dari word embedding dan positional encoding.
        Bagian ini menunjukkan bagaimana kedua komponen tersebut digabungkan untuk membentuk representasi final yang digunakan oleh model.
        """)
        
        # Get word embeddings and positional encoding
        if 'vocabulary' not in locals() or 'all_epoch_embeddings' not in locals():
            # Generate word embeddings if not already done
            with st.spinner("Menghasilkan word embeddings..."):
                vocabulary, all_epoch_embeddings = generate_word_embeddings(
                    vocab_size=vocab_size, 
                    embedding_dim=embedding_dim, 
                    num_epochs=num_epochs
                )
        
        # Generate positional encoding if not already done
        if 'pos_encoding' not in locals():
            pos_encoding = generate_positional_encoding(max_seq_len=100, d_model=embedding_dim)
        
        # Select epoch for word embeddings
        selected_epoch = st.select_slider(
            "Pilih Epoch untuk Word Embeddings", 
            options=list(range(0, num_epochs + 1)),
            value=num_epochs  # Default to final epoch
        )
        
        # Select words and positions
        col1, col2 = st.columns(2)
        with col1:
            selected_words = st.multiselect(
                "Pilih kata untuk divisualisasikan", 
                vocabulary,
                default=["he", "she", "his", "her"]
            )
        
        with col2:
            selected_positions = st.multiselect(
                "Pilih posisi untuk divisualisasikan", 
                list(range(1, 11)),  # Show positions 1-10 for simplicity
                default=[1, 2, 3]
            )
            # Convert to 0-indexed
            selected_positions = [p-1 for p in selected_positions]
        
        # Visualization of the combination process
        with st.expander("Proses Kombinasi Word Embedding dan Positional Encoding", expanded=True):
            if selected_words:
                # Select a word to visualize the combination process
                selected_word = st.selectbox("Pilih kata untuk melihat proses kombinasi", selected_words)
                selected_position = st.selectbox("Pilih posisi untuk melihat proses kombinasi", 
                                               [p+1 for p in selected_positions],  # Show 1-indexed for user
                                               index=0)
                selected_position = selected_position - 1  # Convert back to 0-indexed
                
                # Get the embedding for the selected word
                word_idx = vocabulary.index(selected_word)
                word_embedding = all_epoch_embeddings[selected_epoch][word_idx]
                
                # Visualize the combination process
                combination_fig = visualize_embedding_combination(
                    selected_word, 
                    word_embedding, 
                    pos_encoding, 
                    selected_position
                )
                st.plotly_chart(combination_fig, use_container_width=True)
                
                st.markdown("""
                **Penjelasan Proses Kombinasi:**
                - Grafik menunjukkan nilai untuk setiap dimensi dari word embedding, positional encoding, dan representasi gabungan
                - Representasi gabungan adalah hasil penjumlahan word embedding dan positional encoding
                - Perhatikan bagaimana positional encoding mempengaruhi representasi akhir kata
                """)
        
        # 3D visualization of combined embeddings
        with st.expander("Visualisasi 3D Representasi Gabungan", expanded=True):
            if selected_words and selected_positions:
                # Create 3D visualization
                combined_fig = visualize_combined_embeddings_3d(
                    vocabulary,
                    all_epoch_embeddings[selected_epoch],
                    pos_encoding,
                    selected_positions,
                    highlight_words=selected_words
                )
                st.plotly_chart(combined_fig, use_container_width=True)
                
                st.markdown("""
                **Penjelasan Visualisasi 3D:**
                - Titik biru menunjukkan representasi kata asli (tanpa positional encoding)
                - Titik berwarna lain menunjukkan representasi kata pada posisi yang berbeda
                - Perhatikan bagaimana posisi yang berbeda menghasilkan representasi yang berbeda untuk kata yang sama
                - Kata-kata yang disorot ditandai dengan warna merah
                """)
                
                # Add information about the effect of position on meaning
                st.info("""
                **Mengapa Positional Encoding Penting?**
                
                Dalam model Transformer, urutan kata sangat penting untuk memahami makna kalimat. 
                Positional encoding memungkinkan model untuk membedakan kata yang sama pada posisi yang berbeda dalam kalimat.
                
                Contoh: "Kucing mengejar tikus" vs "Tikus mengejar kucing" - kata-kata yang sama tetapi urutan berbeda menghasilkan makna yang berbeda.
                """)
        
        # Heatmap visualization of combined embeddings
        with st.expander("Heatmap Representasi Gabungan", expanded=True):
            if selected_words and selected_positions:
                # Select a position for heatmap
                heatmap_position = st.selectbox(
                    "Pilih posisi untuk heatmap", 
                    [p+1 for p in selected_positions],  # Show 1-indexed for user
                    index=0
                )
                heatmap_position = heatmap_position - 1  # Convert back to 0-indexed
                
                # Get embeddings for selected words
                word_indices = [vocabulary.index(word) for word in selected_words if word in vocabulary]
                word_embeddings = all_epoch_embeddings[selected_epoch][word_indices]
                
                # Combine with positional encoding
                combined_embeddings = combine_embeddings_with_position(
                    word_embeddings, 
                    pos_encoding, 
                    [heatmap_position]
                )[0]  # Get the first (and only) position's embeddings
                
                # Create heatmap data
                max_dims = min(20, embedding_dim)  # Show at most 20 dimensions
                heatmap_data = np.vstack([
                    word_embeddings[:, :max_dims],  # Original embeddings
                    combined_embeddings[:, :max_dims]  # Combined embeddings
                ])
                
                # Create labels
                heatmap_labels = [
                    f"{word} (original)" for word in selected_words
                ] + [
                    f"{word} (pos {heatmap_position+1})" for word in selected_words
                ]
                
                # Create heatmap
                heatmap_fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Dimensi", y="Kata", color="Nilai"),
                    x=[f"Dim {i+1}" for i in range(max_dims)],
                    y=heatmap_labels,
                    color_continuous_scale="RdBu_r",
                    title=f"Perbandingan Word Embeddings Asli dan Gabungan (Posisi {heatmap_position+1})"
                )
                
                # Update layout
                heatmap_fig.update_layout(
                    height=400 + 20 * len(selected_words)  # Adjust height based on number of words
                )
                
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                st.markdown("""
                **Penjelasan Heatmap:**
                - Baris atas menunjukkan nilai word embedding asli untuk setiap kata
                - Baris bawah menunjukkan nilai representasi gabungan (word embedding + positional encoding)
                - Perhatikan perubahan pola warna yang menunjukkan bagaimana positional encoding mempengaruhi representasi
                """)
                
        # Explanation of the importance in Transformer models
        with st.expander("Pentingnya dalam Model Transformer", expanded=True):
            st.markdown("""
            ### Peran Representasi Gabungan dalam Model Transformer
            
            Dalam arsitektur Transformer, representasi gabungan dari word embedding dan positional encoding menjadi input untuk layer self-attention. Ini memungkinkan model untuk:
            
            1. **Memahami Konteks Kata**: Kata yang sama dapat memiliki makna berbeda tergantung pada posisinya dalam kalimat
            
            2. **Menangkap Hubungan Jarak Jauh**: Self-attention dapat menghubungkan kata-kata yang berjauhan dalam kalimat
            
            3. **Memproses Paralel**: Tidak seperti RNN, Transformer memproses semua kata secara bersamaan, meningkatkan efisiensi
            
            4. **Fleksibilitas Panjang Sekuens**: Model dapat menangani kalimat dengan panjang yang bervariasi
            
            Visualisasi di atas menunjukkan bagaimana representasi gabungan memberikan informasi posisional yang penting sambil mempertahankan informasi semantik dari word embedding asli.
            """)
            
            # Add an illustrative example
            st.image("https://jalammar.github.io/images/t/transformer_positional_encoding_vectors.png", 
                    caption="Ilustrasi Positional Encoding dalam Transformer (Sumber: Jay Alammar)",
                    use_column_width=True)

# Run the application
if __name__ == "__main__":
    main()