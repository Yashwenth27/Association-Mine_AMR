import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)

# Navbar HTML
navbar_html = '''
<style>
    .st-emotion-cache-12fmjuu{
        z-index: 100;
    }
    .st-emotion-cache-h4xjwg{
        z-index: 100;
    }
    h2{
    color: white;
    }
    .css-hi6a2p {padding-top: 0rem;}
    .navbar {
        background-color: #355E3B;
        padding: 0.3rem;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar .logo {
        display: flex;
        align-items: center;
    }
    .navbar .logo img {
        height: 40px;
        margin-right: 10px;
    }
    .navbar .menu {
        display: flex;
        gap: 1.5rem;
    }
    .navbar .menu a {
        color: white;
        font-size: 1.2rem;
        text-decoration: none;
    }
    .content {
        padding-top: 5rem;  /* Adjust this based on navbar height */
    }
</style>

<nav class="navbar">
    <div class="logo">
        <h2 id="hh">MERIT Dashboard</h2>
    </div>
    <div class="menu">
        <a href="">Dashboard</a>
        <a href="mvp">Multi Variable Plots</a>
        <a href="mlp">ML Predictions</a>
        <a href="ml-predictions">About Us</a>
    </div>
</nav>

<div class="content">
'''

# Injecting the navigation bar and content padding into the Streamlit app
st.markdown(navbar_html, unsafe_allow_html=True)

a,b = st.columns([0.3,0.7])

mapping = {
        "Escherichia coli": ["./Final_Ecoli1.csv","./Final_Ecoli2.csv"],
        "Enterobacter spp": "./Final_EB.csv",
        "Enterococcus faecium": "./Final_EF.csv",
        "Klebsiella pneumoniae": "./Final_KP.csv",
        "Acinetobacter baumannii": "./Final_AB.csv",
        "Staphylococcus aureus": ["./Final_SA1.csv","./Final_SA2.csv"]
    }

def getdf(organism):
    if type(mapping[organism])==type([1,2]):
        import pandas as pd
        def combine_csv(file1, file2):
            # Load the two CSV files
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            # Concatenate the two dataframes to get the original dataframe
            combined_df = pd.concat([df1, df2], ignore_index=True)
            
            return combined_df

        # Usage
        original_df = combine_csv(mapping[organism][0], mapping[organism][1])
        return original_df
    else:
        import pandas as pd
        df = pd.read_csv(mapping[organism])
        return df

def plot1(rtor_df):
    arows = rtor_df["antecedents_list"]
    crows = rtor_df["consequents_list"]

    # Create a directed graph
    G = nx.DiGraph()

    # Create a dictionary to map edges to their rules
    edge_rule_map = {}

    # Add edges from antecedents to consequents and map edges to rules
    for index, (antec, conseq) in enumerate(zip(arows, crows)):
        for a in antec:
            for c in conseq:
                G.add_edge(a, c)
                edge_rule_map[(a, c)] = f"If Resistant to {a}, it is resistant to {c}"

    # Positioning nodes using spring layout
    pos = nx.spring_layout(G)

    # Create a list for the edge trace
    edge_trace = go.Scatter(
        x=(),
        y=(),
        line=dict(width=1, color='grey'),
        hoverinfo='text',
        text=(),
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['text'] += tuple([edge_rule_map.get(edge, '')])

    # Create a node trace
    node_trace = go.Scatter(
        x=(),
        y=(),
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='sunset',
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=4
        ),
        textfont=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        textposition="bottom center"
    )

    # Add nodes to the plot with hover text indicating the source (antecedent or consequent)
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        if any(node in a for a in arows):
            source = 'Antecedents'
        else:
            source = 'Consequents'
        node_trace['text'] += tuple([f'{node[0:len(node)-4]}'])
        node_trace['marker']['color'] += tuple([len(list(G.neighbors(node)))])

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph for R to R',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    # Display the graph in Streamlit
    st.plotly_chart(fig)

def plot2(rtos_df):
    arows = rtos_df["antecedents_list"]
    crows = rtos_df["consequents_list"]

    # Create a directed graph
    G = nx.DiGraph()

    # Create a dictionary to map edges to their rules
    edge_rule_map = {}

    # Add edges from antecedents to consequents and map edges to rules
    for index, (antec, conseq) in enumerate(zip(arows, crows)):
        for a in antec:
            for c in conseq:
                G.add_edge(a, c)
                edge_rule_map[(a, c)] = f"If Resistant to {a}, it is resistant to {c}"

    # Positioning nodes using spring layout
    pos = nx.spring_layout(G)

    # Create a list for the edge trace
    edge_trace = go.Scatter(
        x=(),
        y=(),
        line=dict(width=1, color='grey'),
        hoverinfo='text',
        text=(),
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
        edge_trace['text'] += tuple([edge_rule_map.get(edge, '')])

    # Create a node trace
    node_trace = go.Scatter(
        x=(),
        y=(),
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='sunset',
            color=[],
            size=25,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=4
        ),
        textfont=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        textposition="bottom center"
    )

    # Add nodes to the plot with hover text indicating the source (antecedent or consequent)
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        if any(node in a for a in arows):
            source = 'Antecedents'
        else:
            source = 'Consequents'
        node_trace['text'] += tuple([f'{node[0:len(node)-4]}'])
        node_trace['marker']['color'] += tuple([len(list(G.neighbors(node)))])

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network Graph for R to S',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        annotations=[dict(
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    # Display the graph in Streamlit
    st.plotly_chart(fig)

def set_whole(org, b):
    import pandas as pd
    import numpy as np

    
    SA = getdf(org)
    print(SA.head())
    if org!="Escherichia coli":
        lift = st.slider("Choose Lift value (Default @ 2.0)", min_value=0.0, max_value=2.0, step=0.1,value=2.0)
        minsup = st.slider("Choose Minimum Support value (Default <= 0.3)", min_value=0.0, max_value=1.0, step=0.1,value=0.1)
    else:
        lift = st.slider("Choose Lift value (Default @ 2.0)", min_value=0.0, max_value=2.0, step=0.1,value=2.0)
        minsup = st.slider("Choose Minimum Support value (Default <= 0.1)", min_value=0.0, max_value=1.0, step=0.1,value=0.1)
    maxlen=3
    c=0

    if st.button("Apply Filters"):
        df_SA = SA[SA.columns[SA.isnull().sum() / SA.shape[0] < 0.8]]
        df_SA_I_cols = list(df_SA.columns[df_SA.columns.str.contains("_I")])
        SA_input = df_SA[df_SA_I_cols]
        SA_input_getdum = pd.get_dummies(SA_input)
        SA_input_getdum = SA_input_getdum.astype(bool)
        SA_df_freq = apriori(SA_input_getdum, min_support=minsup, max_len=maxlen, use_colnames=True, low_memory=True)
        SA_rules = association_rules(SA_df_freq)
        SA_rules['antecedents_list'] = SA_rules['antecedents'].apply(lambda x: list(x))
        SA_rules['consequents_list'] = SA_rules['consequents'].apply(lambda x: list(x))
        SA_rules.to_csv("SA_whole_rules.csv", index=False)
        #copy to all set_*
        st.write(f"Total Rules Generated {SA_rules.shape[0]}")
        with open("SA_whole_rules.csv", "rb") as file:
            st.download_button(
                label="Download all rules",
                data=file,
                file_name="SA_whole_rules.csv",
                mime="text/csv"
            )
            
        

    with b:
        z,x = st.columns(2)
        # with st.expander("All rules"):
        #     st.write(SA_rules)
        # with st.expander("Rules with Lift > 2"):
        #     
        #     st.write(lift_filtered)
        #     st.write(lift_filtered.shape)
        lift_filtered = SA_rules[SA_rules['lift'] > lift]
        def antecedents_only_R(antecedents_list):
            # Check if all elements in antecedents list end with '_R'
            return all(item.endswith('_R') for item in antecedents_list)

        lift_filtered = lift_filtered[lift_filtered['antecedents_list'].apply(antecedents_only_R)]

        # Step 3: Split the rules into "rtor" and "rtos"
        rtor_rows = []
        rtos_rows = []

        def process_consequents(consequents_list):
            if all(item.endswith('_R') for item in consequents_list):
                return 'rtor'
            else:
                return 'rtos'

        # Iterate over the filtered dataframe and split into rtor and rtos
        for index, row in lift_filtered.iterrows():
            rule_type = process_consequents(row['consequents_list'])
            if rule_type == 'rtor':
                rtor_rows.append(row)
            else:
                rtos_rows.append(row)

        # Create new DataFrames rtor and rtos
        rtor_df = pd.DataFrame(rtor_rows)
        rtos_df = pd.DataFrame(rtos_rows)
        with st.expander("R to R"):
            st.write(rtor_df)
        with st.expander("R to S"):
            st.write(rtos_df)
        q, w = st.columns(2)
        with q:
            st.subheader("Network Plot for R to R ")
            #copy to all set_*
            st.caption(f"Total Rules Extracted - {rtor_df.shape[0]}")
            plot1(rtor_df)
        with w:
            st.subheader("Network Plot for R to S")
            #copy to all set_*
            st.caption(f"Total Rules Extracted - {rtos_df.shape[0]}")
            plot2(rtos_df)

def set_country(org, b):
    import pandas as pd
    import numpy as np

    # Load the data
    SA = getdf(org)
    print(SA.head())
    
    # Get user inputs
    selected_country = st.selectbox("Select Country", options=SA['Country'].unique())
    lift = st.slider("Choose Lift value", min_value=0.0, max_value=2.0, step=0.1)
    minsup = st.slider("Choose Minimum Support value", min_value=0.0, max_value=1.0, step=0.1)
    maxlen = st.slider("Choose Maximum Length of Rules", min_value=1, max_value=3, step=1)
    

    if st.button("Apply Filters"):
        # Filter by the selected country
        df_SA = SA[SA.columns[SA.isnull().sum() / SA.shape[0] < 0.8]]
        df_filtered_country_SA = df_SA[df_SA['Country'] == selected_country]
        df_SA_I_cols = list(df_filtered_country_SA.columns[df_SA.columns.str.contains("_I")])
        df_filtered_country_SA = df_filtered_country_SA[df_SA_I_cols]
        
        # Generate frequent itemsets and association rules
        df_filtered_country_SA = pd.get_dummies(df_filtered_country_SA)
        df_filtered_country_SA = df_filtered_country_SA.astype(bool)
        df_freq = apriori(df_filtered_country_SA, min_support=minsup, max_len=maxlen, use_colnames=True, low_memory=True)
        df_rules = association_rules(df_freq)
        df_rules['antecedents_list'] = df_rules['antecedents'].apply(lambda x: list(x))
        df_rules['consequents_list'] = df_rules['consequents'].apply(lambda x: list(x))
        df_rules.to_csv(f"SA_{selected_country}_rules.csv", index=False)
    
    with b:
        z, x = st.columns(2)
        lift_filtered = df_rules[df_rules['lift'] > lift]
        
        def antecedents_only_R(antecedents_list):
            # Check if all elements in antecedents list end with '_R'
            return all(item.endswith('_R') for item in antecedents_list)

        lift_filtered = lift_filtered[lift_filtered['antecedents_list'].apply(antecedents_only_R)]

        # Split the rules into "rtor" and "rtos"
        rtor_rows = []
        rtos_rows = []

        def process_consequents(consequents_list):
            if all(item.endswith('_R') for item in consequents_list):
                return 'rtor'
            else:
                return 'rtos'

        # Iterate over the filtered dataframe and split into rtor and rtos
        for index, row in lift_filtered.iterrows():
            rule_type = process_consequents(row['consequents_list'])
            if rule_type == 'rtor':
                rtor_rows.append(row)
            else:
                rtos_rows.append(row)

        # Create new DataFrames rtor and rtos
        rtor_df = pd.DataFrame(rtor_rows)
        rtos_df = pd.DataFrame(rtos_rows)
        
        with st.expander("R to R"):
            st.write(rtor_df)
        with st.expander("R to S"):
            st.write(rtos_df)
        
        q, w = st.columns(2)
        with q:
            st.subheader("Network Plot for R to R")
            plot1(rtor_df)
        with w:
            st.subheader("Network Plot for R to S")
            plot2(rtos_df)

def set_age(org, b):
    import pandas as pd
    import numpy as np
    import os
    from mlxtend.frequent_patterns import apriori, association_rules

    # Load the data
    SA = getdf(org)
    print(SA.head())
    
    # Define filtering sliders
    lift = st.slider("Choose Lift value", min_value=0.0, max_value=2.0, step=0.1)
    minsup = st.slider("Choose Minimum Support value", min_value=0.0, max_value=1.0, step=0.1)
    maxlen = st.slider("Choose Maximum Length of Rules", min_value=1, max_value=3, step=1)
    
    age_group = st.selectbox("Select Age Group", options=SA['Age Group'].unique())
    
    if st.button("Apply Filters"):
        # Filter data by selected age group
        df_SA = SA[SA['Age Group'] == age_group]
        
        # Prepare the data for association rule mining
        df_SA_I_cols = list(df_SA.columns[df_SA.columns.str.contains("_I")])
        df_SA_I_cols.append('Age Group')
        SA_input_age = df_SA[df_SA_I_cols]
        
        # Create a directory for the filtered data if it doesn't exist
        if not os.path.exists('SA_Age_Split/'):
            os.mkdir("SA_Age_Split")
            
        # Save filtered data for the selected age group
        SA_input_age.to_csv(f"SA_Age_Split/SA_{age_group}.csv", index=False)
        
        # Process the data
        dfi = pd.read_csv(f"SA_Age_Split/SA_{age_group}.csv", low_memory=False)
        dfi.drop(['Age Group'], axis=1, inplace=True)
        dfi_c = dfi[dfi.columns[dfi.isnull().sum() / dfi.shape[0] < 0.8]]
        dfi_gd = pd.get_dummies(dfi_c, dtype='bool')
        df_freq = apriori(dfi_gd, min_support=minsup, max_len=maxlen, use_colnames=True, low_memory=True)
        
        # Create a directory for frequency and association rules if it doesn't exist
        if not os.path.exists('SA_freq_Asso_age/'):
            os.mkdir("SA_freq_Asso_age/")
        if os.path.exists(f"SA_freq_Asso_age/{age_group}"):
            import shutil
            shutil.rmtree(f"SA_freq_Asso_age/{age_group}")
        os.mkdir(f"SA_freq_Asso_age/{age_group}")
        
        # Save frequency items
        df_freq.to_csv(f"SA_freq_Asso_age/{age_group}/SA_freq_items.csv", index=False)
        
        # Generate and save association rules
        df_rules = association_rules(df_freq)
        df_rules['antecedents_list'] = df_rules['antecedents'].apply(lambda x: list(x))
        df_rules['consequents_list'] = df_rules['consequents'].apply(lambda x: list(x))
        df_rules.to_csv(f"SA_freq_Asso_age/{age_group}/SA_asso_rules.csv", index=False)
        
        with b:
            z, x = st.columns(2)
            lift_filtered = df_rules[df_rules['lift'] > lift]

            def antecedents_only_R(antecedents_list):
                # Check if all elements in antecedents list end with '_R'
                return all(item.endswith('_R') for item in antecedents_list)

            lift_filtered = lift_filtered[lift_filtered['antecedents_list'].apply(antecedents_only_R)]

            # Split the rules into "rtor" and "rtos"
            rtor_rows = []
            rtos_rows = []

            def process_consequents(consequents_list):
                if all(item.endswith('_R') for item in consequents_list):
                    return 'rtor'
                else:
                    return 'rtos'

            # Iterate over the filtered dataframe and split into rtor and rtos
            for index, row in lift_filtered.iterrows():
                rule_type = process_consequents(row['consequents_list'])
                if rule_type == 'rtor':
                    rtor_rows.append(row)
                else:
                    rtos_rows.append(row)

            # Create new DataFrames rtor and rtos
            rtor_df = pd.DataFrame(rtor_rows)
            rtos_df = pd.DataFrame(rtos_rows)
            
            with st.expander("R to R"):
                st.write(rtor_df)
            with st.expander("R to S"):
                st.write(rtos_df)
            
            q, w = st.columns(2)
            with q:
                st.subheader("Network Plot for R to R")
                plot1(rtor_df)
            with w:
                st.subheader("Network Plot for R to S")
                plot2(rtos_df)

def set_year(org, b):
    import pandas as pd
    import numpy as np
    import os
    import shutil
    from mlxtend.frequent_patterns import apriori, association_rules

    # Load the data
    SA = getdf(org)
    print(SA.head())

    # Define filtering sliders
    lift = st.slider("Choose Lift value", min_value=0.0, max_value=2.0, step=0.1)
    minsup = st.slider("Choose Minimum Support value", min_value=0.0, max_value=1.0, step=0.1)
    maxlen = st.slider("Choose Maximum Length of Rules", min_value=1, max_value=3, step=1)
    
    # Select Year
    year = st.selectbox("Select Year", options=SA['Year'].unique())
    
    if st.button("Apply Filters"):
        # Filter data by selected year
        df_SA = SA[SA['Year'] == year]
        
        # Prepare the data for association rule mining
        df_SA_I_cols = list(df_SA.columns[df_SA.columns.str.contains("_I")])
        df_SA_I_cols.append('Year')
        SA_input_year = df_SA[df_SA_I_cols]
        
        # Create a directory for the filtered data if it doesn't exist
        if not os.path.exists('SA_Year_Split/'):
            os.mkdir("SA_Year_Split")
        
        # Save filtered data for the selected year
        SA_input_year.to_csv(f"SA_Year_Split/SA_{year}.csv", index=False)
        
        # Process the data
        dfi = pd.read_csv(f"SA_Year_Split/SA_{year}.csv", low_memory=False)
        dfi.drop(['Year'], axis=1, inplace=True)
        dfi_c = dfi[dfi.columns[dfi.isnull().sum() / dfi.shape[0] < 0.8]]
        dfi_gd = pd.get_dummies(dfi_c, dtype='bool')
        df_freq = apriori(dfi_gd, min_support=minsup, max_len=maxlen, use_colnames=True, low_memory=True)
        
        # Create a directory for frequency and association rules if it doesn't exist
        if not os.path.exists('SA_freq_Asso_year/'):
            os.mkdir("SA_freq_Asso_year/")
        if os.path.exists(f"SA_freq_Asso_year/{year}"):
            shutil.rmtree(f"SA_freq_Asso_year/{year}")
        os.mkdir(f"SA_freq_Asso_year/{year}")
        
        # Save frequency items
        df_freq.to_csv(f"SA_freq_Asso_year/{year}/SA_freq_items.csv", index=False)
        
        # Generate and save association rules
        df_rules = association_rules(df_freq)
        df_rules['antecedents_list'] = df_rules['antecedents'].apply(lambda x: list(x))
        df_rules['consequents_list'] = df_rules['consequents'].apply(lambda x: list(x))
        df_rules.to_csv(f"SA_freq_Asso_year/{year}/SA_asso_rules.csv", index=False)
        
        with b:
            z, x = st.columns(2)
            lift_filtered = df_rules[df_rules['lift'] > lift]

            def antecedents_only_R(antecedents_list):
                # Check if all elements in antecedents list end with '_R'
                return all(item.endswith('_R') for item in antecedents_list)

            lift_filtered = lift_filtered[lift_filtered['antecedents_list'].apply(antecedents_only_R)]

            # Split the rules into "rtor" and "rtos"
            rtor_rows = []
            rtos_rows = []

            def process_consequents(consequents_list):
                if all(item.endswith('_R') for item in consequents_list):
                    return 'rtor'
                else:
                    return 'rtos'

            # Iterate over the filtered dataframe and split into rtor and rtos
            for index, row in lift_filtered.iterrows():
                rule_type = process_consequents(row['consequents_list'])
                if rule_type == 'rtor':
                    rtor_rows.append(row)
                else:
                    rtos_rows.append(row)

            # Create new DataFrames rtor and rtos
            rtor_df = pd.DataFrame(rtor_rows)
            rtos_df = pd.DataFrame(rtos_rows)
            
            with st.expander("R to R"):
                st.write(rtor_df)
            with st.expander("R to S"):
                st.write(rtos_df)
            
            q, w = st.columns(2)
            with q:
                st.subheader("Network Plot for R to R")
                plot1(rtor_df)
            with w:
                st.subheader("Network Plot for R to S")
                plot2(rtos_df)


with a:
    with st.container(border=True):
        org = st.selectbox("Choose Organism", ["Not Chosen", "Acinetobacter baumannii", "Enterobacter spp",
                                               "Escherichia coli", "Enterococcus faecium", "Klebsiella pneumoniae",
                                               "Pseudomonas aeruginosa", "Staphylococcus aureus"
                                               ], key=1)
        base = st.radio("Choose Mining basis", ["Whole", "Country", "Year", "Age"])
        if base == "Whole" and org!="Not Chosen":
            set_whole(org, b)
        elif base == "Country":
            st.write("---")
            set_country(org,b)
        elif base == "Age":
            st.write("---")
            set_age(org,b)
        else:
            st.write("---")
            set_year(org,b)

