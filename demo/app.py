"""Streamlit demo for generative design architecture."""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import tempfile
import os

from generative_design.config import Config
from generative_design.models import VAEArchitecturalGenerator, GANArchitecturalGenerator
from generative_design.data import FloorPlanGenerator
from generative_design.utils import set_seed, get_device, visualize_floor_plan
from generative_design.metrics import ArchitecturalMetrics


def load_model(model_path: str, config: Config, device: torch.device):
    """Load a trained model."""
    if config.model.model_type == 'vae':
        model = VAEArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            hidden_dims=config.model.hidden_dims,
            beta=config.model.beta,
            kl_annealing=config.model.kl_annealing
        )
    elif config.model.model_type == 'gan':
        model = GANArchitecturalGenerator(
            input_size=config.model.input_size,
            latent_dim=config.model.latent_dim,
            generator_hidden_dims=config.model.hidden_dims,
            discriminator_hidden_dims=config.model.hidden_dims,
            use_spectral_norm=config.model.use_spectral_norm
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def generate_samples(model, num_samples: int, device: torch.device, model_type: str):
    """Generate samples from the model."""
    with torch.no_grad():
        if model_type == 'vae':
            generated = model.sample(num_samples, device)
        elif model_type == 'gan':
            generated = model.sample(num_samples, device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert to numpy and denormalize
        generated_np = generated.cpu().numpy()
        generated_np = generated_np * 5  # Denormalize to room types
        generated_np = np.round(generated_np).astype(np.int32)
        
        return generated_np


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Generative Design for Architecture",
        page_icon="🏗️",
        layout="wide"
    )
    
    st.title("🏗️ Generative Design for Architecture")
    st.markdown("Generate architectural floor plans using neural networks")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["vae", "gan"],
        help="Choose between VAE (Variational Autoencoder) or GAN (Generative Adversarial Network)"
    )
    
    # Model file upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Model Checkpoint",
        type=['pth', 'pt', 'ckpt'],
        help="Upload a trained model checkpoint"
    )
    
    # Generation parameters
    st.sidebar.header("Generation Parameters")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=16,
        value=4,
        help="Number of floor plans to generate"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Random seed for reproducible generation"
    )
    
    # Room parameters
    st.sidebar.header("Room Parameters")
    
    width = st.sidebar.slider("Floor Plan Width", 16, 64, 32)
    height = st.sidebar.slider("Floor Plan Height", 16, 64, 32)
    min_rooms = st.sidebar.slider("Minimum Rooms", 2, 10, 3)
    max_rooms = st.sidebar.slider("Maximum Rooms", 3, 15, 8)
    room_types = st.sidebar.slider("Room Types", 2, 10, 5)
    
    # Main content
    if uploaded_file is not None:
        # Load model
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                model_path = tmp_file.name
            
            # Load configuration
            config = Config()
            config.model.model_type = model_type
            config.model.input_size = (height, width)
            config.data.width = width
            config.data.height = height
            config.data.min_rooms = min_rooms
            config.data.max_rooms = max_rooms
            config.data.room_types = room_types
            
            # Set seed
            set_seed(seed)
            
            # Get device
            device = get_device()
            
            # Load model
            with st.spinner("Loading model..."):
                model = load_model(model_path, config, device)
            
            st.success(f"Model loaded successfully on {device}")
            
            # Generation section
            st.header("🎨 Generate Floor Plans")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("Generate New Floor Plans", type="primary"):
                    with st.spinner("Generating floor plans..."):
                        # Generate samples
                        samples = generate_samples(model, num_samples, device, model_type)
                        
                        # Display samples
                        cols = st.columns(min(num_samples, 4))
                        for i, sample in enumerate(samples):
                            with cols[i % 4]:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.imshow(sample, cmap='viridis', origin='upper')
                                ax.set_title(f"Floor Plan {i+1}")
                                ax.axis('off')
                                st.pyplot(fig)
                        
                        # Store samples in session state for analysis
                        st.session_state['generated_samples'] = samples
            
            with col2:
                st.subheader("Quick Actions")
                if st.button("Generate Single Sample"):
                    with st.spinner("Generating..."):
                        sample = generate_samples(model, 1, device, model_type)[0]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.imshow(sample, cmap='viridis', origin='upper')
                        ax.set_title("Generated Floor Plan")
                        ax.axis('off')
                        st.pyplot(fig)
            
            # Analysis section
            if 'generated_samples' in st.session_state:
                st.header("📊 Analysis")
                
                samples = st.session_state['generated_samples']
                
                # Calculate metrics
                metrics = ArchitecturalMetrics(room_types)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Room Statistics")
                    for i, sample in enumerate(samples):
                        stats = metrics.calculate_room_statistics(sample)
                        st.write(f"**Floor Plan {i+1}:**")
                        st.write(f"- Rooms: {stats['total_rooms']}")
                        st.write(f"- Coverage: {stats['coverage_ratio']:.2%}")
                        st.write(f"- Avg Room Size: {stats['avg_room_size']:.1f}")
                
                with col2:
                    st.subheader("Connectivity & Compactness")
                    for i, sample in enumerate(samples):
                        conn = metrics.calculate_connectivity(sample)
                        comp = metrics.calculate_compactness(sample)
                        st.write(f"**Floor Plan {i+1}:**")
                        st.write(f"- Connections: {conn['total_connections']}")
                        st.write(f"- Compactness: {comp['compactness']:.2f}")
                        st.write(f"- Efficiency: {comp['efficiency']:.2%}")
                
                # Diversity analysis
                if len(samples) > 1:
                    st.subheader("Diversity Analysis")
                    diversity = metrics.calculate_diversity(samples)
                    st.write(f"- Diversity Score: {diversity['diversity']:.3f}")
                    st.write(f"- Uniqueness: {diversity['uniqueness']:.2%}")
                    st.write(f"- Unique Plans: {diversity['num_unique']}/{diversity['total_samples']}")
            
            # Clean up temporary file
            os.unlink(model_path)
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Please make sure you're uploading a valid model checkpoint file.")
    
    else:
        # Demo mode with synthetic data
        st.header("🎮 Demo Mode")
        st.info("Upload a model checkpoint to generate floor plans, or try the demo with synthetic data below.")
        
        # Generate synthetic data for demo
        if st.button("Generate Synthetic Floor Plans"):
            with st.spinner("Generating synthetic floor plans..."):
                generator = FloorPlanGenerator(
                    width=width,
                    height=height,
                    min_rooms=min_rooms,
                    max_rooms=max_rooms,
                    min_room_size=3,
                    max_room_size=8,
                    room_types=room_types
                )
                
                samples = generator.generate_dataset(num_samples)
                
                # Display samples
                cols = st.columns(min(num_samples, 4))
                for i, sample in enumerate(samples):
                    with cols[i % 4]:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(sample, cmap='viridis', origin='upper')
                        ax.set_title(f"Synthetic Floor Plan {i+1}")
                        ax.axis('off')
                        st.pyplot(fig)
                
                # Analysis
                st.subheader("📊 Synthetic Data Analysis")
                metrics = ArchitecturalMetrics(room_types)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Room Statistics:**")
                    for i, sample in enumerate(samples):
                        stats = metrics.calculate_room_statistics(sample)
                        st.write(f"Plan {i+1}: {stats['total_rooms']} rooms, {stats['coverage_ratio']:.1%} coverage")
                
                with col2:
                    st.write("**Diversity:**")
                    diversity = metrics.calculate_diversity(samples)
                    st.write(f"Diversity: {diversity['diversity']:.3f}")
                    st.write(f"Uniqueness: {diversity['uniqueness']:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Generative Design for Architecture** - Neural network-based architectural layout generation")


if __name__ == "__main__":
    main()
