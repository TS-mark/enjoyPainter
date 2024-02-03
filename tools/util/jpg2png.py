from PIL import Image

def convert_jpg_to_png(input_path, output_path):
    try:
        # 打开JPEG图像
        with Image.open(input_path) as img:
            # 保存为PNG格式
            img.save(output_path, 'PNG')
            print(f'{input_path} 转换成功为 {output_path}')
    except Exception as e:
        print(f'转换失败: {e}')

# 指定输入和输出路径
input_image_path = 'test1.jpg'
output_image_path = 'test1.png'

# 执行转换
convert_jpg_to_png(input_image_path, output_image_path)
