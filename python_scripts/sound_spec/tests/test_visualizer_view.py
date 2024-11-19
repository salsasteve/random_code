import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from sound_spec.visualizer_view import VisualizerView
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.samplerate = 44100
    model.chunk_size = 512
    model.num_bins = 8
    model.bin_config = [(86, 172), (172, 344), (344, 688), (688, 1376), (1376, 2752), (2752, 5504), (5504, 11008), (11008, 20000)]
    model.create_octave_bins.return_value = model.bins
    model.get_next_chunk.return_value = np.ones(512)
    model.compute_fft.return_value = (np.linspace(0, 22050, 256), np.random.rand(256))
    model.smooth_bins.return_value = np.random.rand(8)
    model.scale_bins.return_value = np.random.randint(0, 64, 8)
    return model

def test_init(mock_model):
    view = VisualizerView(mock_model)
    assert view is not None
    assert view.model is not None
    assert view.num_bins == 8



def test_init_bar_plot(mock_model):
    view = VisualizerView(mock_model)
    view.init_bar_plot()
    assert view.fig is not None
    assert view.ax is not None
    assert len(view.bars) == view.num_bins

def test_init_line_plot(mock_model):
    view = VisualizerView(mock_model)
    view.init_line_plot()
    assert view.fig is not None
    assert view.ax is not None
    assert view.line is not None

def test_update_bar_plot(mock_model):
    view = VisualizerView(mock_model)
    view.init_bar_plot()
    bars = view.update_bar_plot(None)
    assert bars is not None


def test_update_bar_plot_chunk_end(mock_model):
    view = VisualizerView(mock_model)
    view.init_bar_plot()
    view.model.get_next_chunk.return_value = None
    view.ani = MagicMock()
    bars = view.update_bar_plot(None)

    assert bars == (view.bars,)

@patch('sound_spec.visualizer_view.FuncAnimation')
@patch('sound_spec.visualizer_view.plt.show')
def test_animate(mock_func_animation, mock_show, mock_model):
    view = VisualizerView(mock_model)
    view.init_bar_plot()
    anim = view.animate()  # Assign the animation to a variable to keep it in scope

    

    # Verify that FuncAnimation was called with the correct interval
    # expected_interval = (mock_model.chunk_size / mock_model.samplerate) * 1000
    
    # mock_func_animation.assert_called_once_with(
    #     view.fig,
    #     view.update_bar_plot,
    #     init_func=lambda: (view.bars,),
    #     interval=expected_interval,
    #     blit=False
    # )

    # Verify that plt.show() was called
    mock_show.assert_called_once()

    assert anim is not None
