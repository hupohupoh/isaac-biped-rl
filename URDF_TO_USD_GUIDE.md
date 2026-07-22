# URDF → USD 转换指南

## 源文件

URDF 文件位置：
```
F:\RobotProject\go2\unitree_model\rc_v.2.4.1\urdf\v2.4.1.urdf
```

## 转换步骤

### 1. 修复 URDF 的 mesh 路径

URDF 中的 mesh 引用使用了 ROS package 格式：
```xml
<mesh filename="package://固定标零切割足版本2.SLDASM/meshes/base_link.STL" />
```

需要修改为绝对路径或相对路径，指向实际的 STL 文件位置：
```
F:\RobotProject\go2\unitree_model\rc_v.2.4.1\meshes\
```

### 2. 在 Isaac Sim 中导入

1. 打开 Isaac Sim
2. 菜单栏：**File → Import → URDF**
3. 选择修复后的 `v2.4.1.urdf`
4. 在 Import 对话框中设置：
   - **Import Type**: Direct Import（推荐，避免脚本兼容问题）
   - **Links**: 勾选 **Moveable Base**
   - **Joint Configuration**: 选择 **Stiffness**
   - **Drive Type**: 选择 **Force**
   - 勾选 **Allow Self-Collision**

### 3. 验证导入

在 Isaac Sim 的 Stage 窗口中检查：
- 确认 `base_link` 是 root prim
- 确认所有 12 个关节名正确：
  - `r_leg_pitch_joint`, `r_leg_roll_joint`, `r_leg_yaw_joint`
  - `r_knee_pitch_joint`, `r_ankle_pitch_joint`, `r_ankle_roll_joint`
  - `l_leg_pitch_joint`, `l_leg_roll_joint`, `l_leg_yaw_joint`
  - `l_knee_pitch_joint`, `l_ankle_pitch_joint`, `l_ankle_roll_joint`

### 4. 导出 USD

1. 右键点击 root prim
2. **Export → USD**
3. 导出到：
   ```
   F:\RobotProject\biped_demo\source\biped_demo\biped_demo\tasks\manager_based\biped_demo\assets\robot\usd\biped.usd
   ```

### 5. 验证 USD 路径

确认 `biped.py` 中的路径正确指向 USD 文件：
```python
usd_path=f"{BIPED_MODEL_DIR}/biped.usd"
```

## 关键关节参数（从 URDF 提取）

| 关节 | 类型 | 限位 (lower) | 限位 (upper) | 力矩 (Nm) | 速度 (rad/s) |
|------|------|-------------|-------------|-----------|-------------|
| leg_pitch | revolute | -1.57 | 1.57 | 5 | 10 |
| leg_roll | revolute | -1.57 / -0.5 | 0.5 / 1.57 | 5 | 10 |
| leg_yaw | revolute | -1.57 | 1.57 | 5 | 10 |
| knee_pitch | revolute | -1.57 | 1.57 | 5 | 10 |
| ankle_pitch | revolute | -0.5 | 0.5 | 5 | 10 |
| ankle_roll | revolute | -0.5 | 0.5 | 5 | 10 |

**注意**: `leg_roll` 的限位左右不对称：
- 右腿 (r_leg_roll): lower=-1.57, upper=0.5
- 左腿 (l_leg_roll): lower=-0.5, upper=1.57
