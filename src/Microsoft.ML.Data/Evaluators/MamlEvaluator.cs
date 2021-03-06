// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This interface is used by Maml components (the <see cref="EvaluateCommand"/>, the <see cref="CrossValidationCommand"/> 
    /// and the <see cref="EvaluateTransform"/> to evaluate, print and save the results.
    /// The input <see cref="RoleMappedData"/> to the <see cref="IEvaluator.Evaluate"/> and the <see cref="IEvaluator.GetPerInstanceMetrics"/> methods 
    /// should be assumed to contain only the following column roles: label, group, weight and name. Any other columns needed for 
    /// evaluation should be searched for by name in the <see cref="ISchema"/>.
    /// </summary>
    public interface IMamlEvaluator : IEvaluator
    {
        /// <summary>
        /// Print the aggregate metrics to the console.
        /// </summary>
        void PrintFoldResults(IChannel ch, Dictionary<string, IDataView> metrics);

        /// <summary>
        /// Combine the aggregate metrics from multiple folds and print them to the console. If filename is not null then
        /// also save the results to the specified file. If results are from multiple folds, the file will contain
        /// the average results first, and then each fold result. 
        /// Also handle any custom kinds of custom metrics, such as p/r curves for binary classification, or group summary results
        /// for ranking.
        /// </summary>
        void PrintOverallResults(IChannel ch, string filename, params Dictionary<string, IDataView>[] metrics);

        /// <summary>
        /// Create a data view containing only the columns that are saved as per-instance results by Maml commands.
        /// </summary>
        IDataView GetPerInstanceDataViewToSave(RoleMappedData perInstance);
    }

    /// <summary>
    /// A base class implementation of <see cref="IMamlEvaluator"/>. The <see cref="Evaluate"/> and <see cref="GetPerInstanceMetrics"/>
    /// methods create a new <see cref="RoleMappedData"/> containing all the columns needed for evaluation, and call the corresponding
    /// methods on an <see cref="IEvaluator"/> of the appropriate type.
    /// </summary>
    public abstract class MamlEvaluatorBase : IMamlEvaluator
    {
        public abstract class ArgumentsBase : EvaluateInputBase
        {
            // Standard columns.

            [Argument(ArgumentType.AtMostOnce, HelpText = "Column to use for labels.", ShortName = "lab")]
            public string LabelColumn;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Weight column name.", ShortName = "weight")]
            public string WeightColumn;

            // Score columns.

            [Argument(ArgumentType.AtMostOnce, HelpText = "Score column name.", ShortName = "score")]
            public string ScoreColumn;

            // Stratification columns.

            [Argument(ArgumentType.Multiple, HelpText = "Stratification column name.", ShortName = "strat")]
            public string[] StratColumn;
        }

        public static RoleMappedSchema.ColumnRole Strat = "Strat";
        protected readonly IHost Host;

        protected readonly string ScoreColumnKind;
        protected readonly string ScoreCol;
        protected readonly string LabelCol;
        protected readonly string WeightCol;
        protected readonly string[] StratCols;

        protected abstract IEvaluator Evaluator { get; }

        protected MamlEvaluatorBase(ArgumentsBase args, IHostEnvironment env, string scoreColumnKind, string registrationName)
        {
            Contracts.CheckValue(env, nameof(env));
            Host = env.Register(registrationName);
            ScoreColumnKind = scoreColumnKind;
            ScoreCol = args.ScoreColumn;
            LabelCol = args.LabelColumn;
            WeightCol = args.WeightColumn;
            StratCols = args.StratColumn;
        }

        public Dictionary<string, IDataView> Evaluate(RoleMappedData data)
        {
            data = RoleMappedData.Create(data.Data, GetInputColumnRoles(data.Schema, needStrat: true));
            return Evaluator.Evaluate(data);
        }

        protected IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles(RoleMappedSchema schema, bool needStrat = false, bool needName = false)
        {
            Host.CheckValue(schema, nameof(schema));

            var roles = !needStrat || StratCols == null
                ? Enumerable.Empty<KeyValuePair<RoleMappedSchema.ColumnRole, string>>()
                : StratCols.Select(col => RoleMappedSchema.CreatePair(Strat, col));

            if (needName && schema.Name != null)
                roles = roles.Prepend(RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Name, schema.Name.Name));

            return roles.Concat(GetInputColumnRolesCore(schema));
        }

        /// <summary>
        /// All the input columns needed by an evaluator should be added here.
        /// The base class ipmlementation gets the score column, the label column (if exists) and the weight column (if exists).
        /// Override if additional columns are needed.
        /// </summary>
        protected virtual IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRolesCore(RoleMappedSchema schema)
        {
            // Get the score column information.
            var scoreInfo = EvaluateUtils.GetScoreColumnInfo(Host, schema.Schema, ScoreCol, nameof(ArgumentsBase.ScoreColumn),
                ScoreColumnKind);
            yield return RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, scoreInfo.Name);

            // Get the label column information.
            string lab = EvaluateUtils.GetColName(LabelCol, schema.Label, DefaultColumnNames.Label);
            yield return RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Label, lab);

            var weight = EvaluateUtils.GetColName(WeightCol, schema.Weight, null);
            if (!string.IsNullOrEmpty(weight))
                yield return RoleMappedSchema.CreatePair(RoleMappedSchema.ColumnRole.Weight, weight);
        }

        public virtual IEnumerable<MetricColumn> GetOverallMetricColumns()
        {
            return Evaluator.GetOverallMetricColumns();
        }

        public void PrintFoldResults(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckValue(metrics, nameof(metrics));
            PrintFoldResultsCore(ch, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintConfusionMatrixAndPerFoldResults.
        /// Override if something else is needed.
        /// </summary>
        protected virtual void PrintFoldResultsCore(IChannel ch, Dictionary<string, IDataView> metrics)
        {
            ch.AssertValue(ch);
            ch.AssertValue(metrics);

            IDataView fold;
            if (!metrics.TryGetValue(MetricKinds.OverallMetrics, out fold))
                throw ch.Except("No overall metrics found");

            string weightedMetrics;
            string unweightedMetrics = MetricWriter.GetPerFoldResults(Host, fold, out weightedMetrics);
            if (!string.IsNullOrEmpty(weightedMetrics))
                ch.Info(weightedMetrics);
            ch.Info(unweightedMetrics);
        }

        public void PrintOverallResults(IChannel ch, string filename, params Dictionary<string, IDataView>[] metrics)
        {
            Host.CheckValue(ch, nameof(ch));
            Host.CheckNonEmpty(metrics, nameof(metrics));
            PrintOverallResultsCore(ch, filename, metrics);
        }

        /// <summary>
        /// This method simply prints the overall metrics using EvaluateUtils.PrintOverallMetrics.
        /// Override if something else is needed.
        /// </summary>
        protected virtual void PrintOverallResultsCore(IChannel ch, string filename, Dictionary<string, IDataView>[] metrics)
        {
            ch.AssertNonEmpty(metrics);

            IDataView overall;
            if (!TryGetOverallMetrics(metrics, out overall))
                throw ch.Except("No overall metrics found");

            MetricWriter.PrintOverallMetrics(Host, ch, filename, overall, metrics.Length);
        }

        protected bool TryGetOverallMetrics(Dictionary<string, IDataView>[] metrics, out IDataView overall)
        {
            Host.AssertNonEmpty(metrics);

            if (metrics.Length == 1)
                return metrics[0].TryGetValue(MetricKinds.OverallMetrics, out overall);

            overall = null;
            var overallList = new List<IDataView>();
            for (int i = 0; i < metrics.Length; i++)
            {
                var dict = metrics[i];
                IDataView idv;
                if (!dict.TryGetValue(MetricKinds.OverallMetrics, out idv))
                    return false;

                // Add a fold-name column. We add it as a text column, since it is only used for saving the result summary file.
                // We use the first column in the data view as an input column to the LambdaColumnMapper, because it must have an input.
                // We use DvText.NA as the value of this column since for any stratified row the value will be non empty, so we can uniquely identify
                // the overall row using this column.
                var inputColName = idv.Schema.GetColumnName(0);
                var inputColType = idv.Schema.GetColumnType(0);
                idv = Utils.MarshalInvoke(EvaluateUtils.AddTextColumn<int>, inputColType.RawType, Host,
                    idv, inputColName, MetricKinds.ColumnNames.FoldIndex, inputColType, string.Format("Fold {0}", i), "FoldName");

                overallList.Add(idv);
            }
            overall = AppendRowsDataView.Create(Host, overallList[0].Schema, overallList.ToArray());
            return true;
        }

        public IDataTransform GetPerInstanceMetrics(RoleMappedData scoredData)
        {
            Host.AssertValue(scoredData);

            var schema = scoredData.Schema;
            var dataEval = RoleMappedData.Create(scoredData.Data, GetInputColumnRoles(schema));
            return Evaluator.GetPerInstanceMetrics(dataEval);
        }

        private IDataView WrapPerInstance(RoleMappedData perInst)
        {
            var idv = perInst.Data;

            // Make a list of column names that Maml outputs as part of the per-instance data view, and then wrap
            // the per-instance data computed by the evaluator in a ChooseColumnsTransform.
            var cols = new List<ChooseColumnsTransform.Column>();

            // If perInst is the result of cross-validation and contains a fold Id column, include it.
            int foldCol;
            if (perInst.Schema.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.FoldIndex, out foldCol))
                cols.Add(new ChooseColumnsTransform.Column() { Source = MetricKinds.ColumnNames.FoldIndex });

            // Maml always outputs a name column, if it doesn't exist add a GenerateNumberTransform.
            if (perInst.Schema.Name == null)
            {
                var args = new GenerateNumberTransform.Arguments();
                args.Column = new[] { new GenerateNumberTransform.Column() { Name = "Instance" } };
                args.UseCounter = true;
                idv = new GenerateNumberTransform(Host, args, idv);
                cols.Add(new ChooseColumnsTransform.Column() { Name = "Instance" });
            }
            else
                cols.Add(new ChooseColumnsTransform.Column() { Source = perInst.Schema.Name.Name, Name = "Instance" });

            // Maml outputs the weight column if it exists.
            if (perInst.Schema.Weight != null)
                cols.Add(new ChooseColumnsTransform.Column() { Name = perInst.Schema.Weight.Name });

            // Get the other columns from the evaluator.
            foreach (var col in GetPerInstanceColumnsToSave(perInst.Schema))
                cols.Add(new ChooseColumnsTransform.Column() { Name = col });

            var chooseArgs = new ChooseColumnsTransform.Arguments();
            chooseArgs.Column = cols.ToArray();
            idv = new ChooseColumnsTransform(Host, chooseArgs, idv);
            return GetPerInstanceMetricsCore(idv, perInst.Schema);
        }

        /// <summary>
        /// The perInst dataview contains all a name column (called Instance), the FoldId, Label and Weight columns if
        /// they exist, and all the columns returned by <see cref="GetPerInstanceColumnsToSave"/>.
        /// It should be overridden only if additional processing is needed, such as dropping slots in the "top k scores" column
        /// in the multi-class case.
        /// </summary>
        protected virtual IDataView GetPerInstanceMetricsCore(IDataView perInst, RoleMappedSchema schema)
        {
            return perInst;
        }

        public IDataView GetPerInstanceDataViewToSave(RoleMappedData perInstance)
        {
            Host.CheckValue(perInstance, nameof(perInstance));
            var data = RoleMappedData.Create(perInstance.Data, GetInputColumnRoles(perInstance.Schema, needName: true));
            return WrapPerInstance(data);
        }

        /// <summary>
        /// Returns the names of the columns that should be saved in the per-instance results file. These can include
        /// the columns generated by the corresponding <see cref="IRowMapper"/>, or any of the input columns used by
        /// it. The Name and Weight columns should not be included, since the base class includes them automatically.
        /// </summary>
        protected abstract IEnumerable<string> GetPerInstanceColumnsToSave(RoleMappedSchema schema);
    }
}
