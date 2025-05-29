import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置页面
st.set_page_config(page_title="青年亚文化之新二手经济研究", layout="wide")
st.title("青年亚文化之新二手经济研究可视化报告")

# 创建模拟数据
def create_simulated_data():
    # 定义年龄组及其数量
    age_definitions = [('18-22岁', 130), ('23-28岁', 50), ('29岁及以上', 20)]
    income_categories = ['≤1000元', '1001-3000元', '3001-5000元', '≥5001元']

    # 定义各年龄段的收入分布（人数）
    # 这些数字是根据您的描述估算的，您可以根据需要调整
    income_distribution_by_age = {
        '18-22岁': {
            'counts': [75, 45, 10, 0],  # 对应 income_categories 的人数
            'total': 130
        },
        '23-28岁': {
            'counts': [5, 25, 15, 5],
            'total': 50
        },
        '29岁及以上': {
            'counts': [1, 4, 8, 7],
            'total': 20
        }
    }
    # 验证每个年龄段的收入人数总和是否正确
    for age_cat, dist_spec in income_distribution_by_age.items():
        if sum(dist_spec['counts']) != dist_spec['total']:
            raise ValueError(f"Income counts for {age_cat} do not sum to the group total {dist_spec['total']}.")


    generated_age_groups = []
    generated_genders = []
    generated_incomes = []

    for age_category, group_total_count in age_definitions:
        # 1. 生成年龄数据
        generated_age_groups.extend([age_category] * group_total_count)
        
        # 2. 生成性别数据 (保持每个年龄段内约50/50)
        num_male_in_group = group_total_count // 2
        num_female_in_group = group_total_count - num_male_in_group
        genders_for_group = ['男'] * num_male_in_group + ['女'] * num_female_in_group
        np.random.shuffle(genders_for_group)
        generated_genders.extend(genders_for_group)

        # 3. 生成收入数据 (根据年龄段特定分布)
        current_age_income_spec = income_distribution_by_age[age_category]
        incomes_for_group = []
        for i, income_category_count in enumerate(current_age_income_spec['counts']):
            incomes_for_group.extend([income_categories[i]] * income_category_count)
        
        np.random.shuffle(incomes_for_group) # 打乱当前年龄段内的收入顺序
        generated_incomes.extend(incomes_for_group)
    
    # 参与情况
    participated = ['是']*185 + ['否']*15
    platforms = ['闲鱼']*160 + ['校内二手群']*140 + ['得物']*80 + ['转转']*70 + ['多抓鱼']*40
    participation_freq = ['每周多次']*60 + ['每月2-3次']*80 + ['每2-3个月1次']*30 + ['半年1次']*10 + ['一年1次或更少']*5
    
    # 亚文化参与
    subculture_participant = ['是']*117 + ['否']*68
    community_type = ['潮玩圈']*56 + ['汉服圈']*38 + ['复古科技圈']*23 + ['其他']*68
    identity_expression = ['经常']*60 + ['偶尔']*73 + ['很少']*30 + ['从不']*22
    
    # 消费文化影响 (1-5分)
    value_change_A = np.random.normal(4.53, 0.6, 200).clip(1,5)
    value_change_B = np.random.normal(4.21, 0.7, 200).clip(1,5)
    value_change_C = np.random.normal(3.87, 0.8, 200).clip(1,5)
    
    # 数字技术作用
    algorithm_impact = np.random.choice(['发现商品', '引发计划外购买', '无影响', '其他'], 
                                      p=[0.681/1.5, 0.524/1.5, 0.2/1.5, 0.095/1.5], size=200)
    community_impact = np.random.choice(['极大促进', '一定促进', '无影响', '不清楚'], 
                                      p=[0.4, 0.438, 0.15, 0.012], size=200)
    
    # 创建DataFrame
    data = pd.DataFrame({
        '年龄': generated_age_groups,       # 使用新的年龄数据
        '性别': generated_genders,         # 使用新的性别数据
        '月收入': generated_incomes,         # 使用新的、与年龄相关的收入数据
        '是否参与': participated,
        '常用平台': np.random.choice(platforms, 200),
        '参与频率': np.random.choice(participation_freq, 200),
        '是否亚文化参与者': np.random.choice(subculture_participant, 200),
        '社群类型': np.random.choice(community_type, 200),
        '身份表达': np.random.choice(identity_expression, 200),
        '价值认知变化A': value_change_A,
        '价值认知变化B': value_change_B,
        '价值认知变化C': value_change_C,
        '算法影响': algorithm_impact,
        '社区影响': community_impact
    })
    
    # 添加亚文化商品交易标记
    data['交易亚文化商品'] = data.apply(lambda x: '是' if x['是否亚文化参与者']=='是' and np.random.rand()>0.3 else '否', axis=1)
    
    return data

df = create_simulated_data()

# 侧边栏控制面板
st.sidebar.header("数据筛选")
selected_age = st.sidebar.multiselect("选择年龄组", options=df['年龄'].unique(), default=df['年龄'].unique())
selected_gender = st.sidebar.multiselect("选择性别", options=df['性别'].unique(), default=df['性别'].unique())
subculture_filter = st.sidebar.radio("亚文化参与者", options=['全部', '是', '否'], index=0)

# 应用筛选
filtered_df = df[
    (df['年龄'].isin(selected_age)) &
    (df['性别'].isin(selected_gender))
]

if subculture_filter != '全部':
    filtered_df = filtered_df[filtered_df['是否亚文化参与者'] == subculture_filter]

# 1. 人口统计信息
st.header("一、参与者人口统计特征")
col1, col2, col3 = st.columns(3)

with col1:
    fig = px.pie(filtered_df, names='年龄', title='年龄分布')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.pie(filtered_df, names='性别', title='性别分布', color='性别',
                 color_discrete_map={'男':'#1f77b4','女':'#ff7f0e'})
    st.plotly_chart(fig, use_container_width=True)

with col3:
    income_order = ['≤1000元', '1001-3000元', '3001-5000元', '≥5001元']
    fig = px.bar(filtered_df['月收入'].value_counts().reindex(income_order), 
                title='月收入分布', text_auto=True)
    fig.update_layout(xaxis_title='', yaxis_title='人数')
    st.plotly_chart(fig, use_container_width=True)

# 2. 二手经济参与情况
st.header("二、新二手经济参与情况")
tab1, tab2, tab3 = st.tabs(["参与概况", "平台使用", "参与频率"])

with tab1:
    fig = px.pie(filtered_df, names='是否参与', title='新二手经济参与率',
                 color_discrete_sequence=['#2ca02c', '#d62728'])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    platform_counts = filtered_df['常用平台'].value_counts()
    fig = px.bar(platform_counts, title='常用平台分布', text_auto=True)
    fig.update_layout(xaxis_title='平台', yaxis_title='使用人数')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    freq_order = ['每周多次', '每月2-3次', '每2-3个月1次', '半年1次', '一年1次或更少']
    fig = px.bar(filtered_df['参与频率'].value_counts().reindex(freq_order), 
                title='参与频率分布', text_auto=True)
    fig.update_layout(xaxis_title='', yaxis_title='人数')
    st.plotly_chart(fig, use_container_width=True)

# 3. 亚文化参与特征
st.header("三、亚文化参与特征")
col1, col2 = st.columns(2)

with col1:
    fig = px.pie(filtered_df, names='是否亚文化参与者', title='亚文化参与者比例',
                 color_discrete_sequence=['#9467bd', '#8c564b'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Filter data for subculture participants
    subculture_participants_df = filtered_df[filtered_df['是否亚文化参与者'] == '是']
    
    if not subculture_participants_df.empty:
        comm_counts = subculture_participants_df['社群类型'].value_counts()
        
        if not comm_counts.empty:
            # Explicitly pass 'values' and 'names' from the Series
            fig = px.pie(values=comm_counts.values, 
                         names=comm_counts.index, 
                         title='亚文化社群类型分布',
                         hole=0.3,
                         # To ensure diverse colors if not automatically applied well
                         color_discrete_sequence=px.colors.qualitative.Plotly) 
        else:
            # Handle cases where no data is available after filtering
            st.info("当前筛选条件下，没有亚文化社群类型数据可供显示。")
            fig = go.Figure() # Display an empty figure
    else:
        st.info("当前筛选条件下，没有亚文化参与者为'是'的数据可供显示社群类型。")
        fig = go.Figure() # Display an empty figure
    
    st.plotly_chart(fig, use_container_width=True)

# 4. 消费文化影响
st.header("四、对消费文化的影响")
st.subheader("参与二手经济活动后对商品价值的认知变化")

fig = make_subplots(rows=1, cols=3, subplot_titles=(
    "A. 不再只看重新品，二手商品也有价值",
    "B. 更注重商品背后的故事、情感等附加价值",
    "C. 关注商品的文化内涵"
))

for i, col in enumerate(['价值认知变化A', '价值认知变化B', '价值认知变化C'], 1):
    sub_df = filtered_df.groupby('是否亚文化参与者')[col].mean().reset_index()
    fig.add_trace(
        go.Bar(
            x=sub_df['是否亚文化参与者'],
            y=sub_df[col],
            name=col,
            text=np.round(sub_df[col], 2),
            textposition='auto'
        ),
        row=1, col=i
    )

fig.update_layout(height=500, showlegend=False, yaxis_range=[0,5])
fig.update_yaxes(title_text="评分均值 (1-5分)")
fig.update_xaxes(title_text="是否为亚文化参与者")
st.plotly_chart(fig, use_container_width=True)

# 5. 数字技术作用
st.header("五、数字技术在新二手经济中的作用")
tab1, tab2 = st.tabs(["算法推荐影响", "虚拟社区作用"])

with tab1:
    impact_counts = filtered_df['算法影响'].value_counts()
    fig = px.bar(impact_counts, title='算法推荐对交易的影响', text_auto=True)
    fig.update_layout(xaxis_title='影响类型', yaxis_title='人数')
    st.plotly_chart(fig, use_container_width=True)
    
    # 亚文化参与者对比
    cross_tab = pd.crosstab(filtered_df['是否亚文化参与者'], filtered_df['算法影响'])
    fig = px.bar(cross_tab, barmode='group', title='亚文化参与者 vs 非参与者')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not filtered_df.empty:
        comm_impact_counts = filtered_df['社区影响'].value_counts()
        
        if not comm_impact_counts.empty:
            # Explicitly pass 'values' and 'names' from the Series
            fig = px.pie(values=comm_impact_counts.values, 
                         names=comm_impact_counts.index, 
                         title='虚拟社区对参与的影响',
                         # To ensure diverse colors if not automatically applied well
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        else:
            st.info("当前筛选条件下，没有社区影响数据可供显示。")
            fig = go.Figure() # Display an empty figure
    else:
        st.info("当前筛选条件下，没有数据可供显示社区影响。")
        fig = go.Figure() # Display an empty figure
            
    st.plotly_chart(fig, use_container_width=True)

# 6. 身份表达与社群参与
# 6. 身份表达与社群参与
st.header("六、身份认同与社群参与")

fig = make_subplots(rows=1, cols=2, 
                    specs=[[{'type':'domain'}, {'type':'xy'}]],
                    subplot_titles=("身份表达频率 (筛选后全体)", "亚文化参与者的社群特定身份表达")) # Added subplot titles for clarity

# Left pie chart: Identity expression frequency
# This uses the main filtered_df, which correctly reflects the sidebar selection.
if not filtered_df.empty:
    identity_counts = filtered_df['身份表达'].value_counts()
    if not identity_counts.empty:
        fig.add_trace(
            go.Pie(
                labels=identity_counts.index,
                values=identity_counts.values,
                name="身份表达频率"  # Removed hole for simplicity, can be added back: hole=0.4
            ), row=1, col=1)
    else:
        # If filtered_df is not empty but '身份表达' somehow yields no counts
        fig.add_annotation(text="无身份表达数据", row=1, col=1, showarrow=False)
else:
    # If filtered_df itself is empty due to other filters (e.g., age/gender leading to no results)
    fig.add_annotation(text="无数据进行身份表达分析", row=1, col=1, showarrow=False)


# Right bar chart: Community participation of subculture ('是') participants.
# This part should only be populated if there are '是' (Yes) subculture participants in the current view.
subculture_specific_df = filtered_df[filtered_df['是否亚文化参与者'] == '是']

if not subculture_specific_df.empty:
    expression_by_community = pd.crosstab(subculture_specific_df['社群类型'], 
                                          subculture_specific_df['身份表达'])
    
    if not expression_by_community.empty:
        categories_to_plot = ['经常', '偶尔'] # Define categories you intend to plot from original code

        plot_has_data = False # Flag to check if any bar trace was added
        for category in categories_to_plot:
            if category in expression_by_community.columns: # Check if the column exists
                fig.add_trace(
                    go.Bar(
                        x=expression_by_community.index,
                        y=expression_by_community[category],
                        name=category, # Name for legend
                        text=expression_by_community[category],
                        textposition='auto'
                    ), row=1, col=2)
                plot_has_data = True
        
        if plot_has_data:
             fig.update_yaxes(title_text="人数", row=1, col=2) # Apply axis titles only if plot has data
             fig.update_xaxes(title_text="社群类型", row=1, col=2)
        else:
            # expression_by_community was not empty, but desired categories ('经常', '偶尔') were not present
            fig.add_annotation(text="亚文化参与者数据中未找到<br>'经常'或'偶尔'的身份表达分类", 
                               row=1, col=2, showarrow=False)
    else:
        # subculture_specific_df was not empty, but pd.crosstab resulted in an empty DataFrame.
        # This can happen if, for instance, all subculture participants belong to only one '社群类型'
        # or have only one '身份表达' type, making a 2D crosstab empty or 1D.
        fig.add_annotation(text="亚文化参与者数据不足以<br>进行社群与身份表达的交叉分析", 
                           row=1, col=2, showarrow=False)
else:
    # This case is hit if:
    # 1. Sidebar filter "亚文化参与者" is set to "否".
    # 2. Sidebar filter is "是", but other filters (age/gender) result in no '是' participants.
    fig.add_annotation(text="此图表分析亚文化参与者。<br>当前筛选不包含此类数据，<br>或已选择查看非亚文化参与者。", 
                       row=1, col=2, showarrow=False)

fig.update_layout(
    title_text="身份认同与社群参与", 
    height=500,
    showlegend=True # Ensure legend is generally on; Plotly handles items based on traces added
)
st.plotly_chart(fig, use_container_width=True)

# 关键发现总结
st.header("关键研究发现")
st.markdown("""
1. **亚文化参与者特征**  
   - 青年亚文化参与者占63.2%，潮玩圈和汉服圈为主流社群
   - 71.9%的亚文化参与者通过二手商品表达身份认同

2. **消费价值观变革**  
   - 亚文化参与者更关注商品文化内涵(均值4.53 vs 非参与者2.91)
   - 形成"环保-功利-认同"三元悖论式消费观

3. **平台算法影响**  
   - 算法推荐显著影响亚文化消费(72.3%表示发现新商品)
   - 52.4%因算法产生计划外购买，体现"文化折叠效应"

4. **情感价值创造**  
   - 38.5%认为"情感价值创造"是二手经济核心特征
   - 附带故事的二手商品溢价达20-50%
""")

# 添加数据下载功能
st.sidebar.download_button(
    label="下载数据",
    data=df.to_csv().encode('utf-8'),
    file_name='新二手经济研究数据.csv',
    mime='text/csv'
)
