import torch
import time

def test_mamba_env():
    print("--- 系統環境檢查 ---")
    print(f"PyTorch 版本: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {cuda_available}")
    
    if cuda_available:
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("警告: 未檢測到 CUDA，Mamba 在 CPU 上運行效率極低或可能報錯。")

    print("\n--- Mamba 組件檢查 ---")
    try:
        from mamba_ssm import Mamba
        from mamba_ssm.models.config_mamba import MambaConfig
        print("✅ Mamba 套件載入成功")
        
        # 測試一個簡單的 Forward Pass
        batch, length, dim = 2, 64, 16
        x = torch.randn(batch, length, dim).to("cuda" if cuda_available else "cpu")
        
        model = Mamba(
            d_model=dim, 
            d_state=16, 
            d_conv=4, 
            expand=2
        ).to("cuda" if cuda_available else "cpu")
        
        y = model(x)
        
        assert y.shape == x.shape
        print(f"✅ Mamba Forward Pass 測試成功 (Output shape: {y.shape})")
        
    except ImportError as e:
        print(f"❌ Mamba 載入失敗: {e}")
        print("提示: 請檢查 mamba-ssm 與 causal-conv1d 是否已安裝。")
    except Exception as e:
        print(f"❌ 運行時錯誤: {e}")

    try:
        import causal_conv1d
        print("✅ causal-conv1d (Mamba 核心依賴) 載入成功")
    except ImportError:
        print("❌ 未找到 causal-conv1d，這會顯著降低 Mamba 的運行效率。")

if __name__ == "__main__":
    test_mamba_env()