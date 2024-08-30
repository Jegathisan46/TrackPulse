import streamlit as st
import pandas as pd  # Pandas kütüphanesini içe aktarma
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.ticker import PercentFormatter


#.....
# df = pd.read_csv("Top_Seller_Cleaned_temp_2.csv")  # Doğru CSV dosya yolunu girin
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Top_Seller_Cleaned_temp_2.csv")
    
st.download_button(
    label="Download Current Data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='current_data.csv',
    mime='text/csv',
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2E3A44;  /* Deep Charcoal */
    }
    .custom-text {
        font-size: 24px;
        color: #FFD700;  /* Gold */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# A function to add a logo to the top right corner using HTML and CSS.

def add_logo(image_path, height='50px'):
    st.markdown(
        f"""
        <div style="position: absolute; top: 0; right: 0; padding: 10px;">
            <img src="{image_path}" style="height: {height};">
        </div>
        """,
        unsafe_allow_html=True
    )


# We will use a page manager structure to create the page layout.
def Sales_Rank_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'90px') 
    st.title("Sales Rank Analyses")
    
     # Create the filtered data frame.
    filtered_df = df[df['Categories: Root'].isin(['Tools & Home Improvement', 'Home & Kitchen'])]

    # Calculate the 'Sales Rank: 30 days avg.' averages for each main category for each store.
    store_category_rank_30_avg = filtered_df.groupby(['Store_Name', 'Categories: Root'])['Sales Rank: 30 days avg.'].mean().reset_index()
    
    # Create a custom color scale.
    color_discrete_map = {'Tools & Home Improvement': 'red', 'Home & Kitchen': 'blue'}

    # Determine the column ordering using the category_orders parameter.
    category_orders = {"Categories: Root": ["Tools & Home Improvement", "Home & Kitchen"]}

    # Use Plotly Express to create the graph.
    fig = px.bar(store_category_rank_30_avg,
                 x='Store_Name',
                 y='Sales Rank: 30 days avg.',
                 color='Categories: Root',
                 title='30-Day Average Sales Rank by Stores and Main Categories',
                 labels={'Sales Rank: 30 days avg.': '30-Day Average Sales Rank'},
                 hover_data=['Categories: Root'],
                 color_discrete_map=color_discrete_map,
                 category_orders=category_orders)  # Use the color mapping and ordering.

    # Update the axis labels and the graph design.
    fig.update_layout(xaxis_title='Store Name',
                      yaxis_title='30-Day Average Sales Rank',
                      barmode='group',
                      xaxis={'categoryorder':'total descending'})

    # Display the graph in Streamlit.
    st.plotly_chart(fig)


    # Filter the top 20 sales based on the 'Bought in past month' column.
    top_bought_last_month = df.nlargest(20, 'Bought in past month')

    # Create a barplot with Seaborn.
    plt.figure(figsize=(12, 10))
    bar_plot = sns.barplot(
        data=top_bought_last_month,
        y='ASIN',  # Show ASIN on the Y-axis.
        x='Bought in past month',  # Show the amount bought in the past month on the X-axis.
        hue='Categories: Root',  # Use category information for color differentiation.
        dodge=False,  # Categories will overlap.
        palette='viridis'  #  Use the 'viridis' color palette.
)

    # Set the X and Y axis labels and the title of the chart.
    plt.xlabel('Bought in Past Month')
    plt.ylabel('ASIN')
    plt.title('Top 20 Products Bought in the Last Month by Subcategory and Sales Rank 30 Days Avg')

    # Show the 'Sales Rank: 30 days avg.' value in white at the center of each bar.
    for index, (value, rank) in enumerate(zip(top_bought_last_month['Bought in past month'], top_bought_last_month['Sales Rank: 30 days avg.'])):
        plt.text(value / 2, index, f'{rank:.2f}', color='white', ha='center', va='center')

    # Move the legend to the right.
    plt.legend(title='Rootcategory', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust the layout.
    plt.tight_layout()

    # Embed the chart in Streamlit.
    st.pyplot(plt)
    
    st.write("""In this graph, we can see the 30-day average sales rank values for the top 20 best-selling ASINs of the last month, as well as their root categories. We observe that 17 of these products belong to the 'Home & Kitchen' root category. If we compare the sales rank scores in these two categories, we can say that the values for products belonging to 'Home & Kitchen' are lower, indicating that their sales volumes are much higher.""")
    
    # First, filter rows for only the desired categories.
    filtered_df = df[df['Categories: Root'].isin(['Tools & Home Improvement', 'Home & Kitchen'])]

    # Group by Store_Name and Categories: Root columns and count the number of products in each group.
    store_root_category_counts = filtered_df.groupby(['Store_Name', 'Categories: Root']).size().reset_index(name='Product_Count')

    # Sort the results by Store_Name and Categories: Root.
    store_root_category_counts_sorted = store_root_category_counts.sort_values(by=['Store_Name', 'Categories: Root'])

    # Pivot the data by Categories: Root column to prepare for a stacked bar chart.
    pivot_data = store_root_category_counts_sorted.pivot(index='Store_Name', columns='Categories: Root', values='Product_Count').reset_index()

    # Get the "deep" color palette from Seaborn.
    color_palette = sns.color_palette("deep", len(pivot_data.columns[1:])).as_hex()
    
    # Create the chart with Plotly.
    fig = go.Figure()
    
    for idx, category in enumerate(pivot_data.columns[1:]):
        fig.add_trace(go.Bar(
            x=pivot_data['Store_Name'],
            y=pivot_data[category],
            name=category,
            marker_color=color_palette[idx]
    ))
    
        fig.update_layout(
        title='Number of Products by Main Category in Stores',
        xaxis_title='Store Name',
        yaxis_title='Number of Products',
        barmode='stack',
        template="plotly_white"
)


    # Display the Chart.
    st.plotly_chart(fig)

    
    st.write("""**Analyses:**

Here, the Plotly library was used to visualize multiple datasets at once. In the graph above, stores and their products are plotted according to their root categories and the number of products in these categories.

**Findings:**

The diversity in the number of products offered by Layger, NorthernShipmens, and UnbetatableSale stores immediately caught our eye. Additionally, it was observed that these stores in our dataset generally offer a wide range of products in the Home & Kitchen and Tools & Home Improvement root categories.
 

**Next Steps:**

Market share analysis can be conducted based on the number of products in specific categories.

By analyzing sales data and customer preferences according to categories, stores can develop their marketing and inventory strategies.

**Recommendadions:**

Stores with a high variety of products can gain a competitive advantage by offering their customers a wider range of options.

By increasing the number of products in underrepresented categories, a difference can be made in these niche markets.""")

    
def Product_Review_Analyses():
    # Add logo function
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg", '90px') 
    st.title("Product Review Analysis")
     
    # Create a slider in the sidebar and assign its value to the user_input_slider variable
    user_input_slider = st.sidebar.select_slider(
        'Select Product Count for Review Trend Analysis',
        options=[20, 50, 100, 500, 1000, 'all'],
        value='all'  # Default value
    )

    # Definition of the function to show review trend
    def show_review_trend_2(df_r, user_input):
        # Select columns related to "Reviews"
        review_columns = [col for col in df_r.columns if "Reviews" in col]
        # Create a new DataFrame containing ASIN and "Reviews" related columns
        df_reviews = df_r[['ASIN'] + review_columns].copy()
        # Clean missing values
        df_reviews_cleaned = df_reviews.dropna(subset=review_columns)
        # Calculate the average values for rows with the same ASIN in review-related columns
        df_reviews_grouped = df_reviews_cleaned.groupby('ASIN').mean().reset_index()
        
        # Calculate the review trend score
        df_reviews_grouped['review_trend'] = (
            df_reviews_grouped['Reviews: Review Count - 30 days avg.'] - 
            df_reviews_grouped['Reviews: Review Count - 90 days avg.']
        )
        
        # Scale the review trend score between 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_reviews_grouped['review_trend_scaled'] = scaler.fit_transform(df_reviews_grouped[['review_trend']])
        
        # Select all products or the top-scoring products based on user input
        if user_input == 'all':
            top_products = df_reviews_grouped
            plot_count = len(df_reviews_grouped)
        else:
            top_products = df_reviews_grouped.nlargest(user_input, 'review_trend_scaled')
            plot_count = user_input
        
        top_products = top_products.sort_values(by='review_trend_scaled', ascending=False)
        
        # Plot the scatter plot, showing only the highest scoring products
        fig = px.scatter(top_products.head(plot_count), x='ASIN', y='review_trend_scaled',
                         title=f'Top {plot_count} Products (Review Trend Score-2)',
                         labels={'review_trend_scaled': 'Review Trend Score (Scaled)',
                                 'ASIN': 'Product ID'},
                         height=500, symbol_sequence=['star'])
        fig.update_yaxes(range=[0, 1])
        
        return top_products, fig

    # Use the function to assign return values to variables
    df_r, review_trend_fig = show_review_trend_2(df, user_input=user_input_slider)
    
    # Display the DataFrame and the chart in Streamlit
    st.write("Best Products (Review Trend Score-2)")
    st.plotly_chart(review_trend_fig)
    st.dataframe(df_r.head()) # Show the first 5 rows
    

    st.write("""The graph displays a distribution of approximately 28,900 products in Plotly as a scatter plot, sorted from highest to lowest based on the 'Review Trend Score-2'. The horizontal axis represents the ASINs, while the vertical axis shows the 'Trend Review Score'. As 'MinMaxScaler' is used for scaling, the vertical axis is scaled between 0 and 1. The average value of the scores is identified to be around 0.7. The values at the beginning and the end can be considered as outliers. However, the objective here could be to select and offer for sale the products with high 'Review Ratings' from among the top 100, 500, or 1000 products with the highest scores.""")
    
def Price_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'90px') 
    st.title("Price Analysis")
 
    # Define price ranges and labels
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']
    
    # Define price columns and titles
    price_columns = [
        ('New: Current', 'New Current: Number of Products by Price Ranges'),
        ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
        ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
        ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
    ]

    # Define colors
    colors = ['blue', 'green', 'red', 'purple']

    northern_shipments_data = df[df['Store_Name'] == 'NorthernShipments']
    
    # Create chart
    fig = go.Figure()
    
    for (column, title), color in zip(price_columns, colors):
        fbm_prices = northern_shipments_data[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

    fig.update_layout(barmode='group', height=800, width=1000, title_text="NorthernShipments: Number of Products by Different Price Ranges")
    fig.update_xaxes(title_text="Price Range")
    fig.update_yaxes(title_text="Number of Products")

    # Display chart
    st.plotly_chart(fig)
    st.write("Northern Shipments: Contrary to the general pattern, products in the 0-50 range rank second in this particular store. It has been observed that the range with the most products grouped is 50-100 dollars.")
  
    # Define price ranges and labels again
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']

    # Define price columns and titles again
    price_columns = [
        ('New: Current', 'New Current: Number of Products by Price Ranges'),
        ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
        ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
        ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
    ]

    # Define colors again
    colors = ['blue', 'green', 'red', 'purple']
    
    Layger_data = df[df['Store_Name'] == 'Layger']
    
    # Create chart
    fig = go.Figure()

    for (column, title), color in zip(price_columns, colors):
        fbm_prices = Layger_data[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

    fig.update_layout(barmode='group', height=800, width=1000, title_text="Layger: Number of Products by Different Price Ranges")
    fig.update_xaxes(title_text="Price Ranges")
    fig.update_yaxes(title_text="Number of Products")

    # Display chart
    st.plotly_chart(fig)
    st.write("Layger: When examined on a store-by-store basis, it is found that products in the 0-50 dollar range are the most prevalent even in the Layger store, which has the highest product diversity.")
    
    # Define price ranges and labels again
    price_bins = [0, 50, 100, 300, float('inf')]
    price_labels = ['0-50', '50-100', '100-300', '300+']

    # Define price columns and titles again
    price_columns = [
        ('New: Current', 'New Current: Number of Products by Price Ranges'),
        ('Buy Box: Current', 'Buy Box: Number of Products by Price Ranges'),    
        ('New, 3rd Party FBM: Current', 'FBM: Number of Products by Price Ranges'),
        ('Amazon: Current', 'Amazon: Number of Products by Price Ranges')
    ]

    # Define colors again
    colors = ['blue', 'green', 'red', 'purple']

    # Create chart
    fig = go.Figure()

    for (column, title), color in zip(price_columns, colors):
        fbm_prices = df[column]
        price_groups = pd.cut(fbm_prices, bins=price_bins, labels=price_labels)
        price_counts = price_groups.value_counts()
        fig.add_trace(go.Bar(x=price_counts.index, y=price_counts, name=title, marker=dict(color=color), text=price_counts, textposition='auto'))

    fig.update_layout(barmode='group', height=800, width=1000, title_text="Number of Products by Different Price Ranges")
    fig.update_xaxes(title_text="Price Ranges")
    fig.update_yaxes(title_text="Number of Products")

    # Display chart
    st.plotly_chart(fig)

    st.write("General Analysis: When looking at the entire top seller data, it has been observed that sellers are most successful with products in the 0-50 dollar range.")

def Buy_Box_Price_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'90px') 
    st.title("Buy Box Price Analysis")
    
    # Filter data
    matched_seller_df = df[df['Store_Name'] == df['Buy Box Seller_Seller_Name']]
    matched_fba_df = df[df['Store_Name'] == df['Lowest FBA Seller_Seller_Name']]
    matched_fbm_df = df[df['Store_Name'] == df['Lowest FBM Seller_Seller_Name']]
    
    # Draw a triple bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Comparison Type', y='Positive Feedback %', data=pd.concat([
         pd.DataFrame({'Comparison Type': 'Buy Box Seller', 'Positive Feedback %': matched_seller_df['Buy Box Seller_Positive_Feedback']}),
         pd.DataFrame({'Comparison Type': 'FBA Lowest Seller', 'Positive Feedback %': matched_fba_df['Lowest FBA Seller_Positive_Feedback']}),
         pd.DataFrame({'Comparison Type': 'FBM Lowest Seller', 'Positive Feedback %': matched_fbm_df['Lowest FBM Seller_Positive_Feedback']})
    ]), ci=None)

    plt.xlabel('Comparison Type')
    plt.ylabel('Positive Feedback %')
    plt.title('Positive Feedback Comparison for Matching Sellers')

    # Print the percentage values above the bars with a larger font size
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=17,  # Font size
                color='black', 
                xytext=(0, 6),  # Text position
                textcoords='offset points')

    # Enlarge the x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)

    st.pyplot(plt)
    st.write("""
The analysis is conducted on a store-by-store basis, revealing how many products within the same store own the 'buy box'. For products without the buy box, comparisons have been made primarily between FBA and FBM sellers. These comparisons have been examined through the lens of positive seller feedback.

Results indicate that the highest seller positive feedback, at a rate of 91.2%, belongs to the FBA lowest seller. However, it has been observed that top sellers generally have a high rate of positive feedback. This analysis takes into account the fact that a seller can simultaneously be the buy box seller, FBA lowest seller, and FBM lowest seller. It has been noted, though, that high seller ratings positively influence the likelihood of owning the buy box. This suggests that seller ratings generally reflect a seller's performance and reliability. High seller ratings can represent customer satisfaction and positive feedback, potentially increasing the chances of owning the buy box.

However, it should not be forgotten that other factors mentioned in the analysis (FBA lowest seller, FBM lowest seller, etc.) also influence owning the buy box. Therefore, evaluating a seller's success in owning the buy box requires considering multiple factors.

This situation indicates that feedback and sales performance can vary based on different sales strategies.""")
    
# def Other_Related_Features_Analyses ():
#     add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg",'30px') 
#     st.title("Top 1% Products")
#             # HTML ve CSS kullanarak yazı boyutunu biraz daha küçültme
#     st.markdown("""
#     <style>
#     .moderate-font {
#         font-size:18px;  # Yazı boyutunu 22px olarak ayarla
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#     <div class="moderate-font">Products that fall into the top 1% of their subcategories have been specifically analyzed for Northern Shipments and Layger companies in the Home & Kitchen and Tools & Home Improvement categories, leading to the results presented in the table.:</div>
#     """, unsafe_allow_html=True)

    
    
#     st.image("image2.png", use_column_width=True)
#        # HTML ve CSS kullanarak yazı boyutunu biraz daha küçültme
#     st.markdown("""
#     <style>
#     .moderate-font {
#         font-size:18px;  # Yazı boyutunu 22px olarak ayarla
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     st.markdown("""
#     <div class="moderate-font">While the 'Northern Shipments' company has only 6 products in the top 1% tier of the 'Home & Kitchen' group, the 'Layger' company has a total of 303 products across 103 subcategories. Similarly, in the 'Tools & Home Improvement' category, while 'Northern Shipments' has no products in the top 1% tier, 'Layger' has 214 products across 98 subcategories in the 'Home & Kitchen' group. This is indeed an achievement. The reason being, it’s possible to have many products and still rank lower, but having a large number of products and managing to keep a significant portion of them in the top 1% is an example that should be followed by other oneAMZ sellers.:</div>
#     """, unsafe_allow_html=True)
def Interactions_of_Features_Analyses():
    add_logo("https://facts.net/wp-content/uploads/2023/09/14-surprising-facts-about-amazon-1695565756.jpeg", '90px')
    st.title("Interactions of Features")
    
    # Filter "Home & Kitchen" and "Tools & Home Improvement" categories
    home_kitchen_df = df[df['Categories: Root'] == 'Home & Kitchen']
    tools_home_df = df[df['Categories: Root'] == 'Tools & Home Improvement']
    
    # Group by 'Store_Name' column for each category and calculate statistics
    hk_stats = home_kitchen_df.groupby('Store_Name').agg({
        'Reviews: Review Count': 'mean', 
        'Reviews: Rating': 'mean', 
        'ASIN': 'count'
    }).rename(columns={'ASIN': 'Total Products'}).sort_values(by='Reviews: Review Count', ascending=False).head(20)

    thi_stats = tools_home_df.groupby('Store_Name').agg({
        'Reviews: Review Count': 'mean', 
        'Reviews: Rating': 'mean', 
        'ASIN': 'count'
    }).rename(columns={'ASIN': 'Total Products'}).sort_values(by='Reviews: Review Count', ascending=True).head(20)

    # Create a subplot (vertical pyramid)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.02, subplot_titles=("Home & Kitchen", "Tools & Home Improvement"))

    # Horizontal bars for "Home & Kitchen" (average review count) - Yellow
    fig.add_trace(
        go.Bar(
            y=hk_stats.index,
            x=-hk_stats['Reviews: Review Count'],
            orientation='h',
            name='Average Review Count',
            text=hk_stats['Reviews: Review Count'].round(1).astype(str),
            marker=dict(color='yellow', line=dict(color='black', width=1)),
            offsetgroup=0,
        ),
        row=1, col=1
    )
    
    # Horizontal bars for "Tools & Home Improvement" (average review count) - Yellow
    fig.add_trace(
        go.Bar(
            y=thi_stats.index,
            x=thi_stats['Reviews: Review Count'],
            orientation='h',
            name='Average Review Count',
            text=thi_stats['Reviews: Review Count'].round(1).astype(str),
            marker=dict(color='yellow', line=dict(color='black', width=1)),
            offsetgroup=2,
        ),
        row=1, col=2
    )

    # Horizontal bars for "Home & Kitchen" (total products) - Green
    fig.add_trace(
        go.Bar(
            y=hk_stats.index,
            x=-hk_stats['Total Products'],
            orientation='h',
            name='Total Products',
            text=hk_stats['Total Products'].astype(str),
            marker=dict(color='green', line=dict(color='black', width=1)),
            offsetgroup=1,
        ),
        row=1, col=1
    )

    # Horizontal bars for "Tools & Home Improvement" (total products) - Green
    fig.add_trace(
        go.Bar(
            y=thi_stats.index,
            x=thi_stats['Total Products'],
            orientation='h',
            name='Total Products',
            text=thi_stats['Total Products'].astype(str),
            marker=dict(color='green', line=dict(color='black', width=1)),
            offsetgroup=3,
        ),
        row=1, col=2
    )
    
    # Layout settings
    fig.update_layout(
        title_text='Average Review Count and Total Products - Tornado Plot',
        width=1000,
        height=900,
        barmode='group',
        yaxis=dict(title='Store Name', autorange='reversed'),
        xaxis=dict(title='Average Review Count', showticklabels=True),
        xaxis2=dict(title='Average Review Count'),
        showlegend=True,
        legend=dict(
            x=0.1, 
            y=0.1, 
            orientation='v',  # Horizontal positioning
            traceorder='normal'  # Order by trace
        )
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    st.write("""
    Although there isn't a pre-built 'Tornado' type chart in the Plotly library, we have created one ourselves. On the horizontal axis of the graph, the 'Average Review Count' values extend to the right and left, while the vertical axis displays the top competing sellers. The left side of the graph represents the 'Home & Kitchen' main category, and the right side represents 'Tools & Home Improvement'. Yellow bars indicate the 'Average Review Count', and green bars show the stores' 'Total Products', that is, the number of products. The graph ranks the stores from top to bottom based on their 'Average Review Count' in the 'Home & Kitchen' main category. In other words, the average number of reviews per product has been taken as a measure of success.
    """)

# Add a colorful sidebar title using HTML and CSS
st.sidebar.markdown(
    """
    <h1 style='color: orange;'>Analytics Dashboard</h1>
    """, 
    unsafe_allow_html=True
)

# Sidebar title with increased font size and bold font
st.sidebar.markdown(
    """
    <style>
    .big-font {
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    <div class='big-font'>Top Seller Keepa</div>
    """, 
    unsafe_allow_html=True
)

# Correct sidebar radio options with proper comma separation
page = st.sidebar.radio(
    "",
    ['Sales Rank Analysis', 'Product Review Analysis', 'Price Analysis', 'Buy Box Price Analysis', "Interactions of Features Analysis"]
)

# Main part
if __name__ == "__main__":
    if page == 'Sales Rank Analysis':
        Sales_Rank_Analyses()
    elif page == 'Product Review Analysis':
        Product_Review_Analyses()
    elif page == 'Price Analysis':
        Price_Analyses()
    elif page == 'Buy Box Price Analysis':
        Buy_Box_Price_Analyses()
    # elif page == "Top 1% Product Analysis":
    #     Other_Related_Features_Analyses()
    elif page == "Interactions of Features Analysis":
        Interactions_of_Features_Analyses()
