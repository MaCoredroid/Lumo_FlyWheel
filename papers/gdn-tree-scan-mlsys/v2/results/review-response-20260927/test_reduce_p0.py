"""Negative controls for population and acceptance reduction, CPU only."""
import ast
from dataclasses import dataclass, field
from pathlib import Path
import unittest
from reduce_p0 import delta, metric, validate_acceptance, rates


class PopulationControls(unittest.TestCase):
    def test_extracted_engine_observation_contract(self):
        source=Path(__file__).parent/'source-semantics/current-deployment-image/vllm/v1/spec_decode/metrics.py'
        tree=ast.parse(source.read_text())
        stats=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpecDecodingStats')
        namespace={'dataclass':dataclass,'field':field}
        exec(compile(ast.Module(body=[stats],type_ignores=[]),str(source),'exec'),namespace)
        observed=namespace['SpecDecodingStats'].new(31)
        for accepted in (0,0,1,1,1,2,2,2,3,3):
            observed.observe_draft(31,accepted)
        histogram=validate_acceptance(observed.num_drafts,observed.num_draft_tokens,
            observed.num_accepted_tokens,observed.num_accepted_tokens_per_pos)
        self.assertEqual(histogram,[2,3,3,2]+[0]*28)

    def test_accepted_histogram_conservation(self):
        survival=[8,5,2]+[0]*28
        self.assertEqual(validate_acceptance(10,310,15,survival),[2,3,3,2]+[0]*28)

    def test_padding_not_logical_width(self):
        with self.assertRaisesRegex(ValueError,'physical slots'):
            validate_acceptance(10,270,15,[8,5,2]+[0]*28)

    def test_missing_position_rejected(self):
        with self.assertRaisesRegex(ValueError,'31 physical'):
            validate_acceptance(10,310,15,[8,5,2]+[0]*27)

    def test_nonmonotone_survival_rejected(self):
        with self.assertRaisesRegex(ValueError,'monotone'):
            validate_acceptance(10,310,15,[5,8,2]+[0]*28)

    def test_scalar_position_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'position sum'):
            validate_acceptance(10,310,16,[8,5,2]+[0]*28)

    def test_counter_reset_rejected(self):
        with self.assertRaisesRegex(ValueError,'counter reset'):
            delta({('x',()):10},{('x',()):9},'x')

    def test_one_sided_counter_rejected(self):
        with self.assertRaisesRegex(ValueError,'one-sided'):
            delta({}, {('x',()):9}, 'x', optional=True)

    def test_wrong_model_rejected(self):
        with self.assertRaisesRegex(ValueError,'wrong model'):
            metric({('x',(('model_name','other'),)):1},'x')

    def test_multiple_series_not_silently_summed(self):
        with self.assertRaisesRegex(ValueError,'expected one'):
            metric({('x',(('position','0'),)):1,('x',(('position','1'),)):2},'x')

    def test_pooled_formula_is_not_mean_request_rate(self):
        a={'output_tokens':11,'completed_requests':1,'request_latency_sum_s':2,'ttft_sum_s':1,'agent_elapsed_s':3}
        b={'output_tokens':91,'completed_requests':1,'request_latency_sum_s':4,'ttft_sum_s':1,'agent_elapsed_s':5}
        pooled=rates({k:a[k]+b[k] for k in a})['pooled_post_first_token_tokens_per_s']
        arithmetic=(rates(a)['pooled_post_first_token_tokens_per_s']+rates(b)['pooled_post_first_token_tokens_per_s'])/2
        self.assertEqual(pooled,25)
        self.assertEqual(arithmetic,20)


if __name__=='__main__':
    unittest.main()
