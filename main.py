import streamlit as st
import MLP
import base64

st.title('求人原稿に対する応募数の予測')
st.write('多層パーセプトロンによる予測')

uploaded_file=st.file_uploader("テストデータファイルをアップロード", type='csv')

if uploaded_file is not None :
    ans=MLP.reg(uploaded_file)
    st.write('予測結果')
    st.dataframe(ans)
    csv = ans.to_csv(index=False)  
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="result.csv">download</a>'
    st.markdown(f"予測データをダウンロードする {href}", unsafe_allow_html=True)
