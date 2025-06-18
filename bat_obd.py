import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import os

class EVBatteryAnalyzer:
    def __init__(self, charger_file=None, obd_file=None):
        self.charger_data = None
        self.obd_data = None
        self.analysis_results = {}
        self.combined_figures = []
        
        if charger_file:
            self.load_charger_data(charger_file)
        if obd_file:
            self.load_obd_data(obd_file)
    
    def load_charger_data(self, file_path):
        """Load and preprocess charger data"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Clean up duplicate datetime columns
            if 'datetime' in df.columns and 'datetime.1' in df.columns:
                df['datetime'] = df['datetime.1']
                df.drop(columns=['datetime.1'], inplace=True)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure datetime is in proper format
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Clean up column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Calculate additional metrics
            if 'current' in df.columns and 'voltage' in df.columns:
                df['power'] = df['current'] * df['voltage']
            
            self.charger_data = df.sort_values('datetime')
            return True
        except Exception as e:
            print(f"Error loading charger data: {e}")
            return False
    
    def load_obd_data(self, file_path):
        """Load and preprocess OBD data"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Clean up duplicate datetime columns
            if 'datetime' in df.columns and 'datetime.1' in df.columns:
                df['datetime'] = df['datetime.1']
                df.drop(columns=['datetime.1'], inplace=True)
            
            # Handle timestamp conversion if needed
            if 'timestamp' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    # Handle scientific notation timestamps
                    df['timestamp'] = df['timestamp'].astype(float)
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Clean up column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Calculate additional metrics
            if 'current' in df.columns and 'voltage' in df.columns:
                df['power'] = df['current'] * df['voltage']
            
            self.obd_data = df.sort_values('datetime')
            return True
        except Exception as e:
            print(f"Error loading OBD data: {e}")
            return False
    
    def analyze_data(self):
        """Perform comprehensive analysis of the loaded data"""
        if self.charger_data is not None:
            self._analyze_charger_data()
        if self.obd_data is not None:
            self._analyze_obd_data()
        if self.charger_data is not None and self.obd_data is not None:
            self._compare_charger_obd()
        
        self._generate_summary_report()
        self._generate_all_plots()
    
    def _analyze_charger_data(self):
        """Analyze charger-specific data"""
        df = self.charger_data
        results = {}
        
        # Basic stats
        results['charging_duration'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 60
        results['start_soc'] = df['soc'].iloc[0]
        results['end_soc'] = df['soc'].iloc[-1]
        results['soc_change'] = results['end_soc'] - results['start_soc']
        
        # Voltage analysis
        results['avg_voltage'] = df['voltage'].mean()
        results['voltage_range'] = (df['voltage'].min(), df['voltage'].max())
        
        # Current analysis
        results['avg_current'] = df['current'].mean()
        results['max_current'] = df['current'].max()
        results['current_range'] = (df['current'].min(), df['current'].max())
        
        # Power analysis
        if 'power' in df.columns:
            results['avg_power'] = df['power'].mean()
            results['max_power'] = df['power'].max()
            results['total_energy'] = df['power'].sum() * (df['datetime'].diff().dt.total_seconds().mean() / 3600)
        
        # Charging phases analysis
        if 'current' in df.columns:
            # Identify constant current phase
            cc_phase = df[df['current'] >= 0.9 * df['current'].max()]
            if len(cc_phase) > 0:
                results['cc_duration'] = (cc_phase['datetime'].iloc[-1] - cc_phase['datetime'].iloc[0]).total_seconds() / 60
                results['cc_percentage'] = len(cc_phase) / len(df) * 100
            
            # Identify CV phase (when current starts decreasing while voltage is stable)
            if len(cc_phase) > 0:
                cv_phase = df[df.index > cc_phase.index[-1]]
                if len(cv_phase) > 0:
                    results['cv_duration'] = (cv_phase['datetime'].iloc[-1] - cv_phase['datetime'].iloc[0]).total_seconds() / 60
                    results['cv_percentage'] = len(cv_phase) / len(df) * 100
        
        # Battery health indicators
        if 'voltage' in df.columns and 'current' in df.columns:
            # Calculate internal resistance estimate
            delta_v = df['voltage'].max() - df['voltage'].min()
            delta_i = df['current'].max() - df['current'].min()
            if delta_i > 0:
                results['estimated_internal_resistance'] = delta_v / delta_i
            
            # Calculate charge efficiency
            if 'charged_energy' in df.columns and df['charged_energy'].max() > 0:
                theoretical_energy = (results['soc_change'] / 100) * 75  # Assuming 75kWh battery
                actual_energy = df['charged_energy'].max()
                results['charge_efficiency'] = (theoretical_energy / actual_energy) * 100 if actual_energy > 0 else None
        
        self.analysis_results['charger_analysis'] = results
    
    def _analyze_obd_data(self):
        """Analyze OBD-specific data"""
        df = self.obd_data
        results = {}
        
        if df is None or len(df) == 0:
            return
        
        # Basic stats
        results['duration'] = (df['datetime'].iloc[-1] - df['datetime'].iloc[0]).total_seconds() / 60
        results['start_soc'] = df['soc'].iloc[0]
        results['end_soc'] = df['soc'].iloc[-1]
        results['soc_change'] = results['end_soc'] - results['start_soc']
        
        # Voltage analysis
        results['avg_voltage'] = df['voltage'].mean()
        results['voltage_range'] = (df['voltage'].min(), df['voltage'].max())
        
        # Current analysis
        results['avg_current'] = df['current'].mean()
        results['max_current'] = df['current'].max()
        results['current_range'] = (df['current'].min(), df['current'].max())
        
        # Temperature analysis
        if 'temperature' in df.columns:
            results['avg_temp'] = df['temperature'].mean()
            results['temp_range'] = (df['temperature'].min(), df['temperature'].max())
        
        # Battery health indicators
        if 'voltage' in df.columns and 'current' in df.columns and 'soc' in df.columns:
            # Calculate voltage sag under load
            load_condition = df[df['current'].abs() > 5]  # Significant load condition
            if len(load_condition) > 0:
                results['avg_voltage_under_load'] = load_condition['voltage'].mean()
                results['voltage_sag'] = results['avg_voltage'] - results['avg_voltage_under_load']
            
            # Estimate capacity based on SOC change and current
            if results['duration'] > 0 and results['soc_change'] != 0:
                avg_current = df['current'].mean()
                capacity_estimate = (avg_current * results['duration'] / 60) / (results['soc_change'] / 100)
                results['estimated_battery_capacity'] = capacity_estimate
        
        self.analysis_results['obd_analysis'] = results
    
    def _compare_charger_obd(self):
        """Compare charger and OBD data when both are available"""
        results = {}
        
        if self.charger_data is None or self.obd_data is None:
            return
        
        # Time alignment check
        charger_start = self.charger_data['datetime'].min()
        charger_end = self.charger_data['datetime'].max()
        obd_start = self.obd_data['datetime'].min()
        obd_end = self.obd_data['datetime'].max()
        
        results['time_overlap'] = min(charger_end, obd_end) - max(charger_start, obd_start)
        
        # SOC comparison
        charger_soc_change = self.analysis_results['charger_analysis']['soc_change']
        obd_soc_change = self.analysis_results['obd_analysis']['soc_change']
        results['soc_discrepancy'] = abs(charger_soc_change - obd_soc_change)
        
        # Voltage comparison
        charger_avg_voltage = self.analysis_results['charger_analysis']['avg_voltage']
        obd_avg_voltage = self.analysis_results['obd_analysis']['avg_voltage']
        results['voltage_discrepancy'] = abs(charger_avg_voltage - obd_avg_voltage)
        
        self.analysis_results['comparison_analysis'] = results
    
    def _generate_summary_report(self):
        """Generate a text summary of the analysis"""
        report = []
        
        if 'charger_analysis' in self.analysis_results:
            ca = self.analysis_results['charger_analysis']
            report.append("## Charger Data Analysis")
            report.append(f"- Charging Duration: {ca['charging_duration']:.1f} minutes")
            report.append(f"- SOC Change: {ca['soc_change']:.1f}% ({ca['start_soc']}% to {ca['end_soc']}%)")
            report.append(f"- Average Voltage: {ca['avg_voltage']:.1f}V (Range: {ca['voltage_range'][0]:.1f}V to {ca['voltage_range'][1]:.1f}V)")
            report.append(f"- Average Current: {ca['avg_current']:.1f}A (Max: {ca['max_current']:.1f}A)")
            
            if 'avg_power' in ca:
                report.append(f"- Average Power: {ca['avg_power']/1000:.1f}kW (Max: {ca['max_power']/1000:.1f}kW)")
                report.append(f"- Total Energy Delivered: {ca['total_energy']/1000:.2f}kWh")
            
            if 'cc_duration' in ca:
                report.append(f"- Constant Current Phase: {ca['cc_duration']:.1f} minutes ({ca['cc_percentage']:.1f}% of session)")
            
            if 'cv_duration' in ca:
                report.append(f"- Constant Voltage Phase: {ca['cv_duration']:.1f} minutes ({ca['cv_percentage']:.1f}% of session)")
            
            if 'estimated_internal_resistance' in ca:
                report.append(f"- Estimated Internal Resistance: {ca['estimated_internal_resistance']*1000:.2f}mΩ")
            
            if 'charge_efficiency' in ca:
                report.append(f"- Charge Efficiency: {ca['charge_efficiency']:.1f}%")
        
        if 'obd_analysis' in self.analysis_results:
            oa = self.analysis_results['obd_analysis']
            report.append("\n## OBD Data Analysis")
            report.append(f"- Duration: {oa['duration']:.1f} minutes")
            report.append(f"- SOC Change: {oa['soc_change']:.1f}% ({oa['start_soc']}% to {oa['end_soc']}%)")
            report.append(f"- Average Voltage: {oa['avg_voltage']:.1f}V (Range: {oa['voltage_range'][0]:.1f}V to {oa['voltage_range'][1]:.1f}V)")
            report.append(f"- Average Current: {oa['avg_current']:.1f}A (Max: {oa['max_current']:.1f}A)")
            
            if 'avg_temp' in oa:
                report.append(f"- Average Temperature: {oa['avg_temp']:.1f}°C (Range: {oa['temp_range'][0]:.1f}°C to {oa['temp_range'][1]:.1f}°C)")
            
            if 'voltage_sag' in oa:
                report.append(f"- Voltage Sag Under Load: {oa['voltage_sag']:.2f}V")
            
            if 'estimated_battery_capacity' in oa:
                report.append(f"- Estimated Battery Capacity: {oa['estimated_battery_capacity']:.1f}kWh")
        
        if 'comparison_analysis' in self.analysis_results:
            comp = self.analysis_results['comparison_analysis']
            report.append("\n## Charger-OBD Comparison")
            report.append(f"- Time Overlap: {comp['time_overlap'].total_seconds()/60:.1f} minutes")
            report.append(f"- SOC Discrepancy: {comp['soc_discrepancy']:.1f}%")
            report.append(f"- Voltage Discrepancy: {comp['voltage_discrepancy']:.2f}V")
        
        # Battery health assessment
        report.append("\n## Battery Health Assessment")
        
        health_warnings = []
        health_indicators = []
        
        if 'charger_analysis' in self.analysis_results:
            ca = self.analysis_results['charger_analysis']
            
            if 'estimated_internal_resistance' in ca:
                ir = ca['estimated_internal_resistance']
                if ir > 0.1:  # Example threshold
                    health_warnings.append(f"High internal resistance detected ({ir*1000:.1f}mΩ), which may indicate battery aging.")
                else:
                    health_indicators.append(f"Internal resistance within normal range ({ir*1000:.1f}mΩ).")
            
            if 'charge_efficiency' in ca:
                ce = ca['charge_efficiency']
                if ce < 90:  # Example threshold
                    health_warnings.append(f"Low charge efficiency ({ce:.1f}%), which may indicate battery or charger issues.")
                else:
                    health_indicators.append(f"Good charge efficiency ({ce:.1f}%).")
        
        if 'obd_analysis' in self.analysis_results:
            oa = self.analysis_results['obd_analysis']
            
            if 'voltage_sag' in oa:
                vs = oa['voltage_sag']
                if vs > 5:  # Example threshold
                    health_warnings.append(f"Significant voltage sag under load ({vs:.1f}V), which may indicate battery wear.")
                else:
                    health_indicators.append(f"Voltage sag under load is minimal ({vs:.1f}V).")
            
            if 'estimated_battery_capacity' in oa:
                ec = oa['estimated_battery_capacity']
                if ec < 70:  # Example threshold (assuming nominal is ~75kWh)
                    health_warnings.append(f"Reduced estimated battery capacity ({ec:.1f}kWh), which may indicate capacity degradation.")
                else:
                    health_indicators.append(f"Estimated battery capacity appears normal ({ec:.1f}kWh).")
        
        if health_warnings:
            report.append("\n### Potential Issues:")
            report.extend(health_warnings)
        
        if health_indicators:
            report.append("\n### Positive Indicators:")
            report.extend(health_indicators)
        
        if not health_warnings and not health_indicators:
            report.append("Insufficient data for comprehensive battery health assessment.")
        
        self.analysis_results['summary_report'] = "\n".join(report)
    
    def _generate_all_plots(self):
        """Generate all plots and combine them into a single figure with sections"""
        main_fig = go.Figure()
        
        # Add title
        main_fig.add_annotation(
            text="EV Battery Analysis Report",
            xref="paper", yref="paper",
            x=0.5, y=1.1, showarrow=False,
            font=dict(size=24))
        
        # Create a table from the summary report
        if 'summary_report' in self.analysis_results:
            # Convert the report to a table format
            report_lines = self.analysis_results['summary_report'].split('\n')
            headers = []
            values = []
            
            current_section = ""
            for line in report_lines:
                if line.startswith("##"):
                    current_section = line[3:]
                elif line.startswith("###"):
                    current_section += f" - {line[4:]}"
                elif line.startswith("-"):
                    headers.append(current_section)
                    values.append(line[2:])
            
            report_table = go.Figure(data=[go.Table(
                header=dict(values=['Category', 'Value'],
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[headers, values],
                          fill_color='lavender',
                          align='left'))
            ])
            report_table.update_layout(title="Summary Report")
            self.combined_figures.append(report_table)
        
        # Generate charger plots if available
        if self.charger_data is not None:
            self._generate_charger_plots_combined()
        
        # Generate OBD plots if available
        if self.obd_data is not None:
            self._generate_obd_plots_combined()
        
        # Generate comparison plots if both datasets available
        if self.charger_data is not None and self.obd_data is not None:
            self._generate_comparison_plots_combined()
    
    def _generate_charger_plots_combined(self):
        """Generate charger plots for combined output"""
        df = self.charger_data
        
        # Create subplots for charger data
        charger_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                 subplot_titles=("Voltage During Charging", 
                                                "Current During Charging", 
                                                "State of Charge (SOC) During Charging"))
        
        # Voltage plot
        charger_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['voltage'], name="Voltage", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Current plot
        charger_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['current'], name="Current", line=dict(color='green')),
            row=2, col=1
        )
        
        # SOC plot
        charger_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['soc'], name="SOC", line=dict(color='red')),
            row=3, col=1
        )
        
        # Update layout
        charger_fig.update_layout(
            height=800,
            title_text="Charging Session Analysis",
            hovermode="x unified"
        )
        
        # Update y-axes titles
        charger_fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        charger_fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        charger_fig.update_yaxes(title_text="SOC (%)", row=3, col=1)
        
        self.combined_figures.append(charger_fig)
        
        # Create power plot if available
        if 'power' in df.columns:
            power_fig = px.line(df, x='datetime', y='power', 
                               title="Power During Charging Session",
                               labels={'power': 'Power (W)', 'datetime': 'Time'})
            power_fig.update_traces(line_color='purple')
            self.combined_figures.append(power_fig)
        
        # Create voltage vs current scatter plot
        if 'current' in df.columns and 'voltage' in df.columns and 'soc' in df.columns:
            v_i_fig = px.scatter(df, x='current', y='voltage', color='soc',
                               title="Voltage vs Current with SOC Coloring",
                               labels={'current': 'Current (A)', 'voltage': 'Voltage (V)', 'soc': 'SOC (%)'})
            self.combined_figures.append(v_i_fig)
    
    def _generate_obd_plots_combined(self):
        """Generate OBD plots for combined output"""
        df = self.obd_data
        
        # Create subplots for OBD data
        obd_fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              subplot_titles=("Voltage Over Time", 
                                             "Current Over Time", 
                                             "State of Charge (SOC) Over Time"))
        
        # Voltage plot
        obd_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['voltage'], name="Voltage", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Current plot
        obd_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['current'], name="Current", line=dict(color='green')),
            row=2, col=1
        )
        
        # SOC plot
        obd_fig.add_trace(
            go.Scatter(x=df['datetime'], y=df['soc'], name="SOC", line=dict(color='red')),
            row=3, col=1
        )
        
        # Update layout
        obd_fig.update_layout(
            height=800,
            title_text="OBD Data Analysis",
            hovermode="x unified"
        )
        
        # Update y-axes titles
        obd_fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        obd_fig.update_yaxes(title_text="Current (A)", row=2, col=1)
        obd_fig.update_yaxes(title_text="SOC (%)", row=3, col=1)
        
        self.combined_figures.append(obd_fig)
        
        # Add temperature plot if available
        if 'temperature' in df.columns:
            temp_fig = px.line(df, x='datetime', y='temperature',
                             title="Battery Temperature Over Time",
                             labels={'temperature': 'Temperature (°C)', 'datetime': 'Time'})
            temp_fig.update_traces(line_color='orange')
            self.combined_figures.append(temp_fig)
    
    def _generate_comparison_plots_combined(self):
        """Generate comparison plots for combined output"""
        # Find overlapping time period
        min_time = max(self.charger_data['datetime'].min(), self.obd_data['datetime'].min())
        max_time = min(self.charger_data['datetime'].max(), self.obd_data['datetime'].max())
        
        if min_time >= max_time:
            return  # No overlap
        
        # Filter data to overlapping period
        charger_overlap = self.charger_data[(self.charger_data['datetime'] >= min_time) & 
                                          (self.charger_data['datetime'] <= max_time)]
        obd_overlap = self.obd_data[(self.obd_data['datetime'] >= min_time) & 
                                   (self.obd_data['datetime'] <= max_time)]
        
        # Create voltage comparison plot
        voltage_comp_fig = go.Figure()
        voltage_comp_fig.add_trace(go.Scatter(
            x=charger_overlap['datetime'],
            y=charger_overlap['voltage'],
            name="Charger Voltage",
            line=dict(color='blue')
        ))
        voltage_comp_fig.add_trace(go.Scatter(
            x=obd_overlap['datetime'],
            y=obd_overlap['voltage'],
            name="OBD Voltage",
            line=dict(color='red')
        ))
        voltage_comp_fig.update_layout(
            title="Voltage Comparison (Charger vs OBD)",
            xaxis_title="Time",
            yaxis_title="Voltage (V)",
            hovermode="x unified"
        )
        self.combined_figures.append(voltage_comp_fig)
        
        # Create SOC comparison plot if available in both
        if 'soc' in charger_overlap.columns and 'soc' in obd_overlap.columns:
            soc_comp_fig = go.Figure()
            soc_comp_fig.add_trace(go.Scatter(
                x=charger_overlap['datetime'],
                y=charger_overlap['soc'],
                name="Charger SOC",
                line=dict(color='blue')
            ))
            soc_comp_fig.add_trace(go.Scatter(
                x=obd_overlap['datetime'],
                y=obd_overlap['soc'],
                name="OBD SOC",
                line=dict(color='red')
            ))
            soc_comp_fig.update_layout(
                title="SOC Comparison (Charger vs OBD)",
                xaxis_title="Time",
                yaxis_title="SOC (%)",
                hovermode="x unified"
            )
            self.combined_figures.append(soc_comp_fig)
    
    def save_combined_report(self, output_dir="D:\WORKSPACE\PyModules\EV Battery Data Analysis Tool\output", filename="battery_analysis_report.html"):
        """Save all plots and report as a single HTML file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, filename)
        
        # Create a single HTML file with all figures
        with open(file_path, 'w') as f:
            f.write("<html><head><title>EV Battery Analysis Report</title></head><body>")
            
            # Add title
            f.write("<h1 style='text-align:center'>EV Battery Analysis Report</h1>")
            
            # Add summary table
            if 'summary_report' in self.analysis_results:
                report_lines = self.analysis_results['summary_report'].split('\n')
                f.write("<div style='margin:20px; padding:20px; border:1px solid #ccc; border-radius:5px;'>")
           
            
            # Add all figures
            for fig in self.combined_figures:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
            f.write("</body></html>")
        
        print(f"Saved combined report to {file_path}")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your data files
    analyzer = EVBatteryAnalyzer(
        charger_file=r"D:\WORKSPACE\PyModules\EV Battery Data Analysis Tool\2023051609_charger.csv",
        obd_file=r"D:\WORKSPACE\PyModules\EV Battery Data Analysis Tool\2023051609_OBD.csv"
    )
    
    # Perform analysis
    analyzer.analyze_data()
    
    # Save combined report
    analyzer.save_combined_report()
    
    # Print summary to console
    print(analyzer.analysis_results.get('summary_report', "No analysis results available."))