#!/bin/bash
# 清理磁盘空间脚本

echo "当前磁盘使用情况："
df -h / | grep -v Filesystem

echo ""
echo "清理选项："
echo "1. 清理/tmp目录（约635M）"
echo "2. 删除测试用的checkpoints"
echo "3. 删除中间checkpoints，只保留最终checkpoint"
echo ""

# 清理/tmp目录
echo "清理/tmp目录..."
find /tmp -type f -mtime +7 -delete 2>/dev/null
find /tmp -type d -empty -delete 2>/dev/null

# 删除测试用的checkpoints
echo "删除测试用的checkpoints..."
rm -rf /root/hns/TinyRecursiveModels/checkpoints/Addition-ACT-torch/finetune_addition_step1_nocompile
rm -rf /root/hns/TinyRecursiveModels/checkpoints/Addition-ACT-torch/finetune_addition_step1_test
rm -rf /root/hns/TinyRecursiveModels/checkpoints/Addition-ACT-torch/finetune_test_steps

echo ""
echo "清理完成！"
df -h / | grep -v Filesystem

