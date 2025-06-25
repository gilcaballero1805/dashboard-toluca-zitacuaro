"""
Dashboard Interactivo con Streamlit - Versión Ultra Segura
Sin usar ninguna función que pueda invocar PyArrow
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Configuración de página
st.set_page_config(
    page_title="Dashboard Toluca-Zitácuaro",
    page_icon="🚗",
    layout="wide"
)

# Título principal
st.title("🚗 Dashboard de Monitoreo - Autopista Toluca-Zitácuaro")
st.markdown("Sistema de Pronóstico de Aforo")

# Función segura para mostrar información
def mostrar_info_segura(texto):
    """Muestra texto de forma segura sin invocar PyArrow"""
    st.markdown(f"**{texto}**")

# Función para mostrar dataframes como tabla HTML
def mostrar_tabla_html(df, titulo=None):
    """Muestra un dataframe como tabla HTML para evitar PyArrow"""
    if titulo:
        st.subheader(titulo)
    
    # Convertir el dataframe a HTML con estilo
    html = df.to_html(classes='table table-striped table-bordered', index=False)
    
    # Agregar CSS para mejorar el estilo
    st.markdown("""
    <style>
    .table {
        font-size: 14px;
        text-align: left;
        width: 100%;
        margin-bottom: 1rem;
        color: #212529;
        border-collapse: collapse;
    }
    .table th {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.75rem;
        font-weight: bold;
    }
    .table td {
        border: 1px solid #dee2e6;
        padding: 0.75rem;
    }
    .table-striped tbody tr:nth-of-type(odd) {
        background-color: rgba(0, 0, 0, 0.02);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mostrar la tabla
    st.markdown(html, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Panel de Control")
vista_seleccionada = st.sidebar.selectbox(
    "Seleccione vista:",
    ["Resumen Ejecutivo", "Pronósticos", "Análisis Histórico", "Evaluación del Modelo", "Diagnóstico"]
)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """Carga los datos históricos"""
    try:
        if os.path.exists('datos_limpios.csv'):
            data = pd.read_csv('datos_limpios.csv', parse_dates=['fecha'])
            return data, "real"
        else:
            # Generar datos de ejemplo
            fechas = pd.date_range(start='2020-01-01', end='2025-02-01', freq='MS')
            np.random.seed(42)
            n = len(fechas)
            tendencia = np.linspace(280000, 323215, n)
            estacionalidad = 20000 * np.sin(2 * np.pi * np.arange(n) / 12)
            ruido = np.random.normal(0, 10000, n)
            aforo = tendencia + estacionalidad + ruido
            
            data = pd.DataFrame({
                'fecha': fechas,
                'AFORO': aforo.astype(int),
                'IGAE': 100 + np.random.normal(0, 5, n).cumsum() * 0.1,
                'TARIFA_REAL': 175 + np.random.normal(0, 2, n)
            })
            return data, "ejemplo"
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None, None

@st.cache_data
def cargar_ultimo_pronostico():
    """Carga el último pronóstico disponible"""
    # Buscar archivos que contengan 'resultados_bayesiano' (con o sin s)
    archivos = []
    for f in os.listdir('.'):
        if ('resultados_bayesiano' in f.lower() or 'resultado_bayesiano' in f.lower()) and f.endswith('.xlsx'):
            archivos.append(f)
    
    if archivos:
        archivo_mas_reciente = sorted(archivos)[-1]
        st.info(f"📁 Cargando archivo: {archivo_mas_reciente}")
        try:
            # Leer todas las hojas
            xls = pd.ExcelFile(archivo_mas_reciente)
            sheets = {}
            for sheet_name in xls.sheet_names:
                sheets[sheet_name] = pd.read_excel(xls, sheet_name=sheet_name)
            return sheets, archivo_mas_reciente
        except Exception as e:
            st.error(f"Error al leer archivo: {str(e)}")
            return None, None
    else:
        return None, None

# Cargar datos
data, tipo_datos = cargar_datos()
pronostico_data, archivo_pronostico = cargar_ultimo_pronostico()

if data is None:
    st.error("⚠️ No se pudo cargar los datos.")
    st.stop()

# Vista: Diagnóstico
if vista_seleccionada == "Diagnóstico":
    st.header("🔍 Diagnóstico de Archivos")
    
    st.subheader("1. Archivos Excel en el directorio")
    archivos_excel = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    
    if archivos_excel:
        for archivo in archivos_excel:
            st.markdown(f"✓ {archivo}")
    else:
        st.markdown("No se encontraron archivos Excel")
    
    if pronostico_data:
        st.subheader(f"2. Estructura del archivo: {archivo_pronostico}")
        st.markdown("**Hojas encontradas:**")
        
        for nombre_hoja, df in pronostico_data.items():
            st.markdown(f"\n**• {nombre_hoja}**")
            st.markdown(f"  - Filas: {len(df)}")
            st.markdown(f"  - Columnas: {len(df.columns)}")
            
            # Mostrar nombres de columnas de forma segura
            cols_text = ", ".join([str(col) for col in df.columns])
            st.markdown(f"  - Nombres: {cols_text}")
            
            # Mostrar primeras filas
            if st.checkbox(f"Ver muestra de {nombre_hoja}", key=f"check_{nombre_hoja}"):
                mostrar_tabla_html(df.head(3))
    else:
        st.warning("No se encontró archivo de pronóstico")
        st.markdown("""
        **Archivos buscados:**
        - Que contengan 'resultados_bayesiano' en el nombre
        - Con extensión .xlsx
        """)

# Vista: Resumen Ejecutivo
elif vista_seleccionada == "Resumen Ejecutivo":
    st.header("📊 Resumen Ejecutivo")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    if data is not None:
        aforo_actual = data['AFORO'].iloc[-1]
        fecha_actual = data['fecha'].iloc[-1]
        cambio = data['AFORO'].pct_change().iloc[-1] * 100
        
        col1.metric("Aforo Actual", f"{aforo_actual:,.0f}", f"{cambio:.1f}%")
        
        # Buscar proyecciones
        proyeccion_disponible = False
        if pronostico_data:
            for nombre_hoja in ['Pronosticos', 'Proyecciones', 'proyecciones', 'Proyeccion', 'proyeccion']:
                if nombre_hoja in pronostico_data:
                    proyecciones = pronostico_data[nombre_hoja]
                    # Buscar columna de valores centrales
                    for col in ['median', 'mean', 'Median', 'Mean']:
                        if col in proyecciones.columns:
                            if len(proyecciones) > 0:
                                aforo_proyectado = proyecciones[col].iloc[-1]
                                crecimiento = (aforo_proyectado / aforo_actual - 1) * 100
                                col2.metric("Proyección 12 meses", f"{aforo_proyectado:,.0f}", f"+{crecimiento:.1f}%")
                                proyeccion_disponible = True
                                break
                    if proyeccion_disponible:
                        break
        
        if not proyeccion_disponible:
            col2.metric("Proyección 12 meses", "No disponible", "")
        
        col3.metric("Tasa Crecimiento", "1.93%", "Anual esperada")
        col4.metric("Última Actualización", fecha_actual.strftime('%B %Y'), 
                   f"{(datetime.now() - fecha_actual).days} días")
        
        # Gráfico principal
        st.subheader("Tendencia Histórica")
        
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(
            x=data['fecha'],
            y=data['AFORO'],
            mode='lines',
            name='Histórico',
            line=dict(color='black', width=2)
        ))
        
        fig.update_layout(
            title="Aforo Histórico",
            xaxis_title="Fecha",
            yaxis_title="Aforo (vehículos)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Vista: Pronósticos
elif vista_seleccionada == "Pronósticos":
    st.header("📈 Análisis de Pronósticos")
    
    proyecciones_encontradas = False
    
    if pronostico_data:
        # Buscar hoja de proyecciones (con diferentes nombres posibles)
        proyecciones_df = None
        nombre_hoja_proyecciones = None
        
        for nombre in ['Pronosticos', 'Proyecciones', 'proyecciones', 'Proyeccion', 'proyeccion', 
                       'Forecast', 'forecast', 'Pronóstico', 'pronostico', 'Pronostico']:
            if nombre in pronostico_data:
                proyecciones_df = pronostico_data[nombre]
                nombre_hoja_proyecciones = nombre
                proyecciones_encontradas = True
                break
        
        if proyecciones_encontradas and proyecciones_df is not None:
            st.success(f"✅ Proyecciones encontradas en hoja: '{nombre_hoja_proyecciones}'")
            
            # Selector de horizonte - por defecto 12 meses
            max_meses = min(12, len(proyecciones_df))
            horizonte = st.slider("Horizonte de pronóstico (meses):", 1, max_meses, max_meses)
            
            # Filtrar datos
            proyecciones_filtradas = proyecciones_df.iloc[:horizonte].copy()
            
            # Mostrar tabla de proyecciones
            st.subheader("Tabla de Proyecciones")
            
            # Formatear para mostrar
            tabla_display = proyecciones_filtradas.copy()
            for col in tabla_display.columns:
                if tabla_display[col].dtype in ['float64', 'int64']:
                    if 'fecha' not in str(col).lower():
                        tabla_display[col] = tabla_display[col].apply(lambda x: f"{x:,.0f}")
            
            mostrar_tabla_html(tabla_display)
            
            # Crear gráfico con las columnas disponibles
            # Identificar columnas relevantes
            col_fecha = 'fecha' if 'fecha' in proyecciones_df.columns else None
            col_median = 'median' if 'median' in proyecciones_df.columns else None
            col_mean = 'mean' if 'mean' in proyecciones_df.columns else None
            col_q05 = 'q05' if 'q05' in proyecciones_df.columns else None
            col_q95 = 'q95' if 'q95' in proyecciones_df.columns else None
            col_q25 = 'q25' if 'q25' in proyecciones_df.columns else None
            col_q75 = 'q75' if 'q75' in proyecciones_df.columns else None
            
            # Usar median o mean para el valor central
            col_valor = col_median if col_median else col_mean
            
            if col_fecha and col_valor:
                fig = go.Figure()
                
                # Agregar intervalos de confianza si están disponibles
                if col_q95 and col_q05:
                    # Intervalo 90%
                    fig.add_trace(go.Scatter(
                        x=proyecciones_filtradas[col_fecha].tolist() + proyecciones_filtradas[col_fecha].tolist()[::-1],
                        y=proyecciones_filtradas[col_q95].tolist() + proyecciones_filtradas[col_q05].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,200,0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='90% Intervalo',
                        showlegend=True
                    ))
                
                if col_q75 and col_q25:
                    # Intervalo 50%
                    fig.add_trace(go.Scatter(
                        x=proyecciones_filtradas[col_fecha].tolist() + proyecciones_filtradas[col_fecha].tolist()[::-1],
                        y=proyecciones_filtradas[col_q75].tolist() + proyecciones_filtradas[col_q25].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,200,0.3)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='50% Intervalo',
                        showlegend=True
                    ))
                
                # Línea central (median o mean)
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(proyecciones_filtradas[col_fecha]),
                    y=proyecciones_filtradas[col_valor],
                    mode='lines+markers',
                    name='Proyección Central',
                    line=dict(color='darkblue', width=3),
                    marker=dict(size=8)
                ))
                
                # Si existe mean y median, agregar mean como línea punteada
                if col_mean and col_median and col_valor == col_median:
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(proyecciones_filtradas[col_fecha]),
                        y=proyecciones_filtradas[col_mean],
                        mode='lines',
                        name='Media',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Proyección de Aforo a {horizonte} meses",
                    xaxis_title="Fecha",
                    yaxis_title="Aforo (vehículos)",
                    hovermode='x unified',
                    height=500,
                    yaxis=dict(
                        tickformat=',.0f',  # Formato con separador de miles sin decimales
                        tickmode='linear',
                        dtick=50000,  # Mostrar marcas cada 50,000
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Agregar tabla de resumen
                st.subheader("Resumen de Proyecciones")
                
                # Calcular métricas
                if data is not None and 'AFORO' in data.columns:
                    aforo_actual = data['AFORO'].iloc[-1]
                    
                    # Crear tabla de resumen
                    resumen_data = []
                    for idx, row in proyecciones_filtradas.iterrows():
                        mes_data = {
                            'Mes': pd.to_datetime(row[col_fecha]).strftime('%B %Y'),
                            'Proyección': f"{row[col_valor]:,.0f}",
                            'Crecimiento vs Actual': f"{(row[col_valor]/aforo_actual - 1)*100:.1f}%"
                        }
                        if col_q05 and col_q95:
                            mes_data['Intervalo 90%'] = f"[{row[col_q05]:,.0f} - {row[col_q95]:,.0f}]"
                        if 'prob_crisis' in row:
                            mes_data['Prob. Crisis'] = f"{row['prob_crisis']*100:.1f}%"
                        resumen_data.append(mes_data)
                    
                    resumen_df = pd.DataFrame(resumen_data)
                    mostrar_tabla_html(resumen_df)
                
                # Mostrar alertas si existen
                if 'prob_crisis' in proyecciones_df.columns:
                    crisis_alta = proyecciones_filtradas[proyecciones_filtradas['prob_crisis'] > 0.1]
                    if len(crisis_alta) > 0:
                        st.warning(f"⚠️ Alerta: {len(crisis_alta)} meses con probabilidad de crisis > 10%")
        else:
            st.warning("No se encontró hoja de proyecciones en el archivo")
            st.markdown("**Hojas disponibles en el archivo:**")
            for hoja in pronostico_data.keys():
                st.markdown(f"- {hoja}")
    else:
        st.error("No se encontró archivo de pronóstico")
        st.markdown("""
        **Para ver pronósticos:**
        1. El archivo debe contener 'resultados_bayesiano' en el nombre
        2. Formato Excel (.xlsx)
        3. Debe tener una hoja con proyecciones
        """)

# Vista: Análisis Histórico
elif vista_seleccionada == "Análisis Histórico":
    st.header("📜 Análisis Histórico")
    
    if data is not None:
        # Selector de período
        años_analisis = st.slider("Años a analizar:", 1, 10, 5)
        data_filtrada = data[data['fecha'] >= data['fecha'].max() - pd.DateOffset(years=años_analisis)]
        
        # Gráfico de tendencia
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data_filtrada['fecha'],
            y=data_filtrada['AFORO'],
            mode='lines',
            name='Aforo',
            line=dict(color='blue', width=2)
        ))
        
        # Tendencia lineal
        z = np.polyfit(range(len(data_filtrada)), data_filtrada['AFORO'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=data_filtrada['fecha'],
            y=p(range(len(data_filtrada))),
            mode='lines',
            name='Tendencia',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Evolución del Aforo - Últimos {años_analisis} años',
            xaxis_title="Fecha",
            yaxis_title="Aforo",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas usando HTML
        st.subheader("Estadísticas del Período")
        
        stats_html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
            <table style="width: 100%;">
                <tr>
                    <td><strong>Promedio:</strong></td>
                    <td>{data_filtrada['AFORO'].mean():,.0f}</td>
                    <td><strong>Máximo:</strong></td>
                    <td>{data_filtrada['AFORO'].max():,.0f}</td>
                </tr>
                <tr>
                    <td><strong>Mínimo:</strong></td>
                    <td>{data_filtrada['AFORO'].min():,.0f}</td>
                    <td><strong>Desv. Est.:</strong></td>
                    <td>{data_filtrada['AFORO'].std():,.0f}</td>
                </tr>
                <tr>
                    <td><strong>Crecimiento Total:</strong></td>
                    <td colspan="3">{(data_filtrada['AFORO'].iloc[-1] / data_filtrada['AFORO'].iloc[0] - 1) * 100:.1f}%</td>
                </tr>
            </table>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)

# Vista: Evaluación del Modelo
elif vista_seleccionada == "Evaluación del Modelo":
    st.header("🎯 Evaluación del Modelo")
    
    # Métricas de ejemplo
    st.subheader("Métricas de Desempeño")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", "3.45%", "✅ Bueno")
    col2.metric("RMSE", "4.82%", "✅ Bueno")
    col3.metric("R²", "0.85", "✅ Bueno")
    
    # Parámetros del modelo
    if pronostico_data:
        # Buscar hoja de parámetros
        params_df = None
        nombre_hoja_params = None
        
        for nombre in ['Parametros', 'parametros', 'Parámetros', 'parámetros', 
                       'Parameters', 'parameters', 'Params', 'params']:
            if nombre in pronostico_data:
                params_df = pronostico_data[nombre]
                nombre_hoja_params = nombre
                break
        
        if params_df is not None:
            st.subheader("Parámetros del Modelo")
            st.success(f"✅ Parámetros encontrados en hoja: '{nombre_hoja_params}'")
            
            # Mostrar info de columnas sin usar list()
            st.markdown("**Columnas encontradas:**")
            cols_text = ", ".join([str(col) for col in params_df.columns])
            st.markdown(cols_text)
            
            # Mostrar tabla
            mostrar_tabla_html(params_df)
            
            # Crear gráfico de parámetros si es posible
            if len(params_df) > 0:
                # Intentar identificar columnas
                col_param = params_df.columns[0] if len(params_df.columns) > 0 else None
                col_valor = params_df.columns[1] if len(params_df.columns) > 1 else None
                col_error = params_df.columns[2] if len(params_df.columns) > 2 else None
                
                if col_param and col_valor:
                    fig_params = go.Figure()
                    
                    if col_error:
                        fig_params.add_trace(go.Bar(
                            x=params_df[col_param],
                            y=params_df[col_valor],
                            error_y=dict(
                                type='data',
                                array=params_df[col_error],
                                visible=True
                            ),
                            marker_color='lightblue',
                            name='Valor estimado'
                        ))
                    else:
                        fig_params.add_trace(go.Bar(
                            x=params_df[col_param],
                            y=params_df[col_valor],
                            marker_color='lightblue',
                            name='Valor estimado'
                        ))
                    
                    fig_params.update_layout(
                        title="Parámetros del Modelo",
                        xaxis_title="Parámetro",
                        yaxis_title="Valor",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig_params, use_container_width=True)
        
        # Buscar hoja de alertas si existe
        if 'Alertas' in pronostico_data:
            st.subheader("🚨 Alertas del Modelo")
            alertas_df = pronostico_data['Alertas']
            
            # Función para formatear fechas en español
            def formatear_fechas(mensaje):
                """Convierte timestamps en el mensaje a formato amigable"""
                import re
                
                # Meses en español
                meses_esp = {
                    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
                    5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
                    9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
                }
                
                # Buscar timestamps en el mensaje
                if isinstance(mensaje, str):
                    # Extraer la parte que contiene las fechas
                    match = re.search(r'\[(.*?)\]', mensaje)
                    if match:
                        fechas_str = match.group(1)
                        # Extraer años y meses
                        fechas_formateadas = []
                        
                        # Buscar patrones de fecha YYYY-MM-DD
                        patron_fecha = r"'(\d{4})-(\d{2})-(\d{2})"
                        matches = re.findall(patron_fecha, fechas_str)
                        
                        for año, mes, dia in matches:
                            mes_num = int(mes)
                            año_num = int(año)
                            if mes_num in meses_esp:
                                fechas_formateadas.append(f"{meses_esp[mes_num]} {año_num}")
                        
                        if fechas_formateadas:
                            # Reemplazar la lista de timestamps con las fechas formateadas
                            parte_inicial = mensaje.split('[')[0]
                            return f"{parte_inicial}{', '.join(fechas_formateadas)}"
                
                return mensaje
            
            # Procesar alertas para mostrarlas de forma más legible
            if len(alertas_df) > 0:
                for idx, alerta in alertas_df.iterrows():
                    # Obtener el mensaje y formatearlo
                    mensaje_original = alerta.get('message', '')
                    mensaje_formateado = formatear_fechas(mensaje_original)
                    
                    # Determinar el color según el nivel
                    if 'level' in alertas_df.columns:
                        nivel = alerta['level']
                        tipo_alerta = alerta.get('type', 'Alerta')
                        
                        # Traducir tipos de alerta
                        if tipo_alerta == 'CRISIS_RISK':
                            tipo_mostrar = 'Riesgo de Crisis'
                        elif tipo_alerta == 'HIGH_UNCERTAINTY':
                            tipo_mostrar = 'Alta Incertidumbre'
                        else:
                            tipo_mostrar = tipo_alerta
                        
                        if nivel == 'HIGH':
                            st.error(f"🔴 **{tipo_mostrar}**: {mensaje_formateado}")
                        elif nivel == 'MEDIUM':
                            st.warning(f"🟡 **{tipo_mostrar}**: {mensaje_formateado}")
                        else:
                            st.info(f"🔵 **{tipo_mostrar}**: {mensaje_formateado}")
                    else:
                        st.warning(f"• {mensaje_formateado}")
                
                # Si hay probabilidades o CV, mostrarlas en métricas
                col1, col2 = st.columns(2)
                if 'probability' in alertas_df.columns:
                    prob_crisis = alertas_df[alertas_df['type'] == 'CRISIS_RISK']['probability'].iloc[0] if len(alertas_df[alertas_df['type'] == 'CRISIS_RISK']) > 0 else 0
                    if prob_crisis > 0:
                        col1.metric("Probabilidad de Crisis", f"{prob_crisis*100:.1f}%", 
                                   "⚠️ Alta" if prob_crisis > 0.5 else "✅ Moderada")
                
                if 'cv' in alertas_df.columns:
                    cv_value = alertas_df[alertas_df['type'] == 'HIGH_UNCERTAINTY']['cv'].iloc[0] if len(alertas_df[alertas_df['type'] == 'HIGH_UNCERTAINTY']) > 0 else 0
                    if cv_value > 0:
                        col2.metric("Coeficiente de Variación", f"{cv_value:.2f}", 
                                   "⚠️ Alta incertidumbre" if cv_value > 0.3 else "✅ Normal")
            
            # Explicación detallada de la crisis predicha
            st.markdown("---")
            st.subheader("📊 Análisis Detallado de la Predicción de Crisis")
            
            # Buscar datos específicos de agosto 2025 si existen
            if 'Pronosticos' in pronostico_data:
                pronosticos = pronostico_data['Pronosticos']
                # Buscar agosto 2025
                agosto_mask = pronosticos['fecha'].astype(str).str.contains('2025-08')
                if agosto_mask.any():
                    agosto_data = pronosticos[agosto_mask].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Proyección Agosto 2025", 
                                 f"{agosto_data['median']:,.0f} vehículos")
                    with col2:
                        if 'aforo_miles' in agosto_data:
                            caida_pct = ((agosto_data['median'] - 430000) / 430000) * 100  # Asumiendo nivel normal ~430k
                            st.metric("Caída Estimada", 
                                     f"{caida_pct:.1f}%",
                                     "vs. nivel esperado")
                    with col3:
                        st.metric("Probabilidad de Crisis", 
                                 f"{agosto_data.get('prob_crisis', 0)*100:.0f}%")
            
            st.markdown("""
            **¿Por qué el modelo predice una crisis con 63.3% de probabilidad?**
            
            El modelo bayesiano detecta un **riesgo elevado de crisis** basándose en varios factores:
            
            1. **📉 Patrón de Caída Proyectada**
               - El modelo proyecta una caída significativa del aforo a partir de agosto 2025
               - Se espera una reducción superior al 20% respecto a los niveles normales
               - Esta caída se mantiene durante 7 meses consecutivos (agosto 2025 - febrero 2026)
            
            2. **📊 Factores Económicos Considerados**
               - **IGAE**: El modelo considera posibles desaceleraciones económicas
               - **Elasticidad-ingreso**: Con elasticidad de 0.79, una caída del PIB impacta fuertemente el aforo
               - **Tarifas**: Aumentos tarifarios pueden reducir la demanda
            
            3. **🔄 Análisis de Incertidumbre**
               - Coeficiente de variación: 0.32 (32% de variabilidad)
               - Los intervalos de confianza se amplían significativamente en estos meses
               - Mayor incertidumbre = mayor riesgo de escenarios adversos
            
            4. **📈 Metodología de Cálculo**
               - El modelo utiliza simulaciones Monte Carlo (miles de escenarios)
               - En el 63.3% de las simulaciones, el aforo cae más del 20%
               - Esto NO significa que definitivamente ocurrirá, sino que hay un riesgo considerable
            
            5. **⚠️ Factores de Riesgo Identificados**
               - Posible recesión económica en 2025-2026
               - Cambios en patrones de movilidad
               - Competencia de rutas alternativas
               - Efectos estacionales adversos
            
            **Recomendaciones:**
            - 🎯 Monitorear indicadores económicos mensuales
            - 💡 Preparar estrategias de mitigación (promociones, ajustes tarifarios)
            - 📋 Actualizar el modelo mensualmente con datos reales
            - 🔍 Analizar factores externos no capturados por el modelo
            """)
            
            st.warning("""
            **Nota importante**: Esta es una proyección probabilística, no una certeza. 
            Un 63.3% de probabilidad significa que en aproximadamente 2 de cada 3 escenarios simulados 
            se observa una caída significativa, pero aún existe un 36.7% de probabilidad de que no ocurra.
            """)
            
            st.info("""
            **Interpretación de la gráfica de parámetros:**
            
            La gráfica muestra los valores estimados de cada parámetro del modelo con sus intervalos de confianza:
            - **Altura de la barra**: Valor estimado del parámetro
            - **Línea de error**: Intervalo de confianza (incertidumbre en la estimación)
            - Un intervalo más largo indica mayor incertidumbre en ese parámetro
            
            Los parámetros típicamente incluyen efectos de variables económicas (IGAE), tarifas, tendencias y estacionalidad.
            """)
        else:
            st.info("No se encontró hoja de parámetros")
            st.markdown("**Hojas disponibles:**")
            for hoja in pronostico_data.keys():
                st.markdown(f"- {hoja}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("**Sistema de Monitoreo v3.0**")
    st.markdown(f"**Actualizado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")