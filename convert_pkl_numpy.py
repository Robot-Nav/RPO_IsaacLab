"""将numpy 2.x pickle文件转换为numpy 1.x兼容格式"""
import pickle
import os
import sys
import io


class CrossVersionUnpickler(pickle.Unpickler):
    """自定义Unpickler：将numpy._core映射为numpy.core"""
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def convert_pkl(src_path, dst_path=None):
    if dst_path is None:
        dst_path = src_path
    print(f"转换: {src_path}")
    with open(src_path, 'rb') as f:
        data = CrossVersionUnpickler(f).load()
    with open(dst_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  -> {dst_path}  完成")


def main():
    data_dirs = [
        "/home/fatu08/roboparty_train/robolab/data/motions/rpo_dance_lab",
        "/home/fatu08/roboparty_train/robolab/data/motions/rpo_dance_gmr",
    ]

    target_file = sys.argv[1] if len(sys.argv) > 1 else None

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.pkl'):
                continue
            if target_file and fname != target_file:
                continue
            src = os.path.join(data_dir, fname)
            convert_pkl(src)


if __name__ == '__main__':
    main()
